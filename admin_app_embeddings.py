import streamlit as st
import os
import io
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
import datetime
import base64  # For embedding serialization
from supabase import create_client, Client

# ────────────────────────────────────────────────
# Page config — MUST BE FIRST
# ────────────────────────────────────────────────
st.set_page_config(page_title="Ontech Employee Manager", layout="wide")

# ────────────────────────────────────────────────
# Supabase client
# ────────────────────────────────────────────────
supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

# Connection check
try:
    supabase.table("employees").select("emp_code", count="planned").limit(0).execute()
    st.sidebar.success("Connected to Supabase ✓")
except Exception as e:
    st.sidebar.error(f"Supabase connection issue: {str(e)}")

# ────────────────────────────────────────────────
# Styling
# ────────────────────────────────────────────────
OCEAN_BLUE   = "#0066cc"
DEEP_BLUE    = "#004080"
LIGHT_BLUE   = "#3A3433"
GREEN_ACCENT = "#2e7d32"
LIGHT_GREEN  = "#05580C"
PURPLE_ACCENT = "#7e57c2"
LIGHT_PURPLE = "#5A1919"
GRAY_BG      = "#0f1316"
TEXT_DARK    = "#1a1a2e"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {GRAY_BG}; }}
    .block-container {{ padding-top: 2rem !important; padding-bottom: 2rem !important; }}
    h1, h2, h3 {{ color: {DEEP_BLUE}; }}
    .stButton > button[kind="primary"] {{
        background-color: {OCEAN_BLUE}; color: white; border: none; border-radius: 6px; padding: 0.6rem 1.2rem;
    }}
    .stButton > button[kind="primary"]:hover {{ background-color: {LIGHT_BLUE}; color: {OCEAN_BLUE}; }}
    hr {{ background-color: {OCEAN_BLUE}; height: 2px; border: none; }}
    .dataframe {{ background-color: white; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }}
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────
BASE_FOLDER = "data"
DATASET_FOLDER = os.path.join(BASE_FOLDER, "dataset")
os.makedirs(DATASET_FOLDER, exist_ok=True)

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if "selected_emp_code" not in st.session_state:
    st.session_state.selected_emp_code = None
if "last_processed" not in st.session_state:
    st.session_state.last_processed = None
if "save_result" not in st.session_state:
    st.session_state.save_result = None
if "save_messages" not in st.session_state:
    st.session_state.save_messages = []

# ────────────────────────────────────────────────
# Face model (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def get_face_model():
    with st.spinner("Loading InsightFace buffalo_s model..."):
        app = FaceAnalysis(name="buffalo_s")
        app.prepare(ctx_id=0, det_size=(640, 640))
    st.success("Model loaded", icon="✅")
    return app

app = get_face_model()

# ────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────
def load_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
        return np.array(img), img
    except Exception as e:
        st.warning(f"Failed to load image: {e}")
        return None, None

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def has_embedding(emp):
    emb = emp.get("embedding")
    if emb is None:
        return "No"
    if isinstance(emb, bytes):
        return f"Yes ({len(emb)} bytes)"
    if isinstance(emb, str):
        return f"Yes (base64)"
    return "Yes"

# ────────────────────────────────────────────────
# Core: process photos → embedding → save to Supabase
# ────────────────────────────────────────────────
def process_employee(emp_code, full_name, department, designation, mobile, notes, uploaded_files):
    messages = []

    emp_code = emp_code.strip().upper()
    if not emp_code:
        st.error("Employee Code is required")
        return messages

    embedding_to_save = None

    if uploaded_files:
        embeddings = []
        emp_folder = os.path.join(DATASET_FOLDER, emp_code)
        os.makedirs(emp_folder, exist_ok=True)

        with st.spinner("Processing up to 3 valid face photos..."):
            valid_count = 0
            for up_file in uploaded_files:
                if valid_count >= 3:  # Limit to 3 valid photos
                    messages.append(f"Stopped at 3 valid photos — {up_file.name} ignored")
                    break

                try:
                    if up_file.size > 5 * 1024 * 1024:
                        messages.append(f"{up_file.name} too large (>5MB) → skipped")
                        continue

                    img_cv, img_pil = load_image(up_file.getvalue())
                    if img_cv is None:
                        messages.append(f"{up_file.name} load failed → skipped")
                        continue

                    faces = app.get(img_cv)
                    if len(faces) != 1:
                        messages.append(f"{up_file.name}: {len(faces)} faces → skipped")
                        continue

                    face = faces[0]
                    if face.det_score < 0.75:
                        messages.append(f"{up_file.name}: low confidence ({face.det_score:.2f}) → skipped")
                        continue

                    emb = face.embedding
                    emb_norm = normalize(emb)
                    
                    # Validate shape (must be 512 for buffalo_s)
                    if emb_norm.shape != (512,):
                        messages.append(f"{up_file.name}: wrong embedding shape {emb_norm.shape} → skipped")
                        continue

                    embeddings.append(emb_norm)
                    valid_count += 1
                    fname = os.path.splitext(up_file.name)[0] + ".png"
                    img_pil.save(os.path.join(emp_folder, fname))
                    messages.append(f"{up_file.name}: valid (conf {face.det_score:.2f})")

                except Exception as e:
                    messages.append(f"Error processing {up_file.name}: {str(e)} → skipped (continuing)")

        if len(embeddings) >= 3:
            emb_stack = np.array(embeddings)  # (3, 512)
            mean_emb = np.mean(emb_stack, axis=0)
            mean_emb_norm = normalize(mean_emb)
            
            if mean_emb_norm.shape == (512,):
                embedding_to_save = mean_emb_norm.tobytes()
                messages.append(f"Mean embedding created: 2048 bytes (512 floats)")
            else:
                messages.append(f"CRITICAL: Mean shape wrong {mean_emb_norm.shape} — no embedding saved")
        elif len(embeddings) > 0:
            messages.append(f"⚠️ Only {len(embeddings)} valid photos (need ≥3 for mean embedding)")
        else:
            messages.append("⚠️ No valid photos → saving without embedding")

    # Save / upsert to Supabase
    with st.spinner("Saving to Supabase..."):
        try:
            data = {
                "emp_code": emp_code,
                "full_name": full_name.strip() or None,
                "department": department.strip() or None,
                "designation": designation.strip() or None,
                "mobile": mobile.strip() or None,
                "notes": notes.strip() or None,
                # Let DB default handle registered_date / updated_at
            }

            # FIXED: Base64-encode embedding to avoid "bytes not JSON serializable"
            if embedding_to_save:
                b64_encoded = base64.b64encode(embedding_to_save).decode('utf-8')
                data["embedding"] = b64_encoded
                messages.append(f"Embedding base64-encoded: {len(b64_encoded)} chars")

            response = supabase.table("employees").upsert(
                data,
                on_conflict="emp_code"
            ).execute()

            messages.append(f"Supabase affected {response.count} row(s)")
            if response.data:
                messages.append("Saved preview: " + str(response.data[0]))

            return messages

        except Exception as e:
            error_str = str(e)
            messages.append(f"Supabase error: {error_str}")
            st.error(f"Save failed: {error_str}")

            if "bytes" in error_str or "serializable" in error_str:
                st.warning("Bytes serialization issue — check base64 encoding")
            if "permission" in error_str.lower():
                st.warning("RLS / permission — disable RLS for testing")
            if "column" in error_str.lower():
                st.warning("Missing column — add embedding / registered_date in Supabase")

            import traceback
            traceback.print_exc()
            return messages

# ────────────────────────────────────────────────
# Main UI & Navigation
# ────────────────────────────────────────────────
st.title("🧑‍💼 Ontech Employee & Attendance Manager")
st.markdown(f"<h3 style='color:{PURPLE_ACCENT};'>Admin Control Panel</h3>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Section",
    ["Main Dashboard (Overview)", "Register / Edit Employee", "Today's Attendance", "Employee Attendance History", "Daily Attendance Report"]
)

# ────────────────────────────────────────────────
# Main Dashboard (with embedding status)
# ────────────────────────────────────────────────
if page == "Main Dashboard (Overview)":
    st.subheader("Registered Employees & Today's Attendance Status")

    try:
        response = supabase.table("employees").select(
            "emp_code, full_name, department, designation, mobile, notes, embedding"
        ).execute()
        employees = response.data or []
    except Exception as e:
        st.error(f"Failed to load employees: {e}")
        employees = []

    if employees:
        present_count = 0  # Placeholder until attendance migrated

        st.caption(f"**Today ({datetime.date.today():%Y-%m-%d})**: {present_count} / {len(employees)} checked in")

        employees_data = []
        for emp in employees:
            code = emp["emp_code"]
            name = emp.get("full_name") or code
            dept = emp.get("department") or ""
            desig = emp.get("designation") or ""
            mob = emp.get("mobile") or ""
            notes = emp.get("notes") or ""

            employees_data.append({
                "Code": code,
                "Name": name,
                "Department": dept,
                "Designation": desig,
                "Mobile": mob,
                "Notes": notes[:100] + "…" if len(notes) > 100 else notes,
                "Has Embedding": has_embedding(emp),
                "Present Today": "—"  # Update when attendance is cloud
            })

        df = pd.DataFrame(employees_data)

        # Highlight embedding status
        def highlight_embedding(row):
            color = LIGHT_GREEN if row["Has Embedding"].startswith("Yes") else "#f8d7da"
            return [f'background-color: {color}' if col == "Has Embedding" else '' for col in df.columns]

        styled_df = df.style.apply(highlight_embedding, axis=1)

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Notes": st.column_config.TextColumn(width="medium"),
                "Has Embedding": st.column_config.TextColumn(width="small")
            }
        )

        if st.button("🔄 Refresh", type="primary"):
            st.rerun()

        # Quick edit buttons
        st.markdown("**Quick Edit:**")
        cols = st.columns(6)
        for i, emp in enumerate(employees_data):
            with cols[i % 6]:
                label = f"✏ {emp['Code']}"
                if emp['Name'] != emp['Code']:
                    label += f" – {emp['Name'][:15]}…"
                if st.button(label, key=f"edit_{emp['Code']}", use_container_width=True):
                    st.session_state.selected_emp_code = emp["Code"]
                    st.rerun()
    else:
        st.info("No employees registered yet. Add someone below.", icon="ℹ️")

# ────────────────────────────────────────────────
# Register / Edit Employee (with photo limit & edit support)
# ────────────────────────────────────────────────
elif page == "Register / Edit Employee":
    st.subheader("Add / Edit Employee")

    emp_code_value = full_name_value = department_value = designation_value = mobile_value = notes_value = ""

    editing = bool(st.session_state.selected_emp_code)

    if editing:
        code = st.session_state.selected_emp_code
        try:
            resp = supabase.table("employees").select("*").eq("emp_code", code).execute()
            if resp.data:
                emp = resp.data[0]
                full_name_value = emp.get("full_name", "") or ""
                department_value = emp.get("department", "") or ""
                designation_value = emp.get("designation", "") or ""
                mobile_value = emp.get("mobile", "") or ""
                notes_value = emp.get("notes", "") or ""
                emp_code_value = code
                st.info(f"Editing employee: **{code}** (add photos to update embedding)", icon="✏️")
        except Exception as e:
            st.error(f"Failed to load employee: {e}")

        if st.button("× Cancel editing", type="secondary"):
            st.session_state.selected_emp_code = None
            st.rerun()

    col1, col2 = st.columns([3, 3])

    with col1:
        emp_code_input = st.text_input("Employee Code (unique)", value=emp_code_value,
                                       key="emp_code_input", disabled=editing)
        full_name = st.text_input("Full Name", value=full_name_value, key="full_name")
        department = st.text_input("Department / Team", value=department_value, key="department")

    with col2:
        designation = st.text_input("Designation / Role", value=designation_value, key="designation")
        mobile = st.text_input("Mobile Number", value=mobile_value, key="mobile")

    notes = st.text_area("Notes / Remarks", value=notes_value, height=110, key="notes")

    st.subheader("Face Photos (up to 3 valid processed, max 5MB each)")
    uploaded_files = st.file_uploader(
        "Upload clear face photos (JPG/PNG only)",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        key="uploader"
    )

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s):")
        cols = st.columns(min(5, len(uploaded_files) or 1))
        for i, file in enumerate(uploaded_files):
            try:
                img = Image.open(file)
                cols[i % len(cols)].image(img, caption=file.name, use_container_width=True)
            except:
                st.warning(f"Cannot preview {file.name}")

    col_btn1, col_btn2 = st.columns([4, 2])

    with col_btn1:
        if st.button("💾 Save / Update Employee", type="primary", use_container_width=True):
            emp_code = (
                st.session_state.selected_emp_code
                if st.session_state.selected_emp_code
                else emp_code_input.strip().upper()
            )

            if not emp_code:
                st.error("Employee Code is required")
            else:
                # Reset save state
                st.session_state.save_result = None
                st.session_state.save_messages = []

                with st.spinner("Processing and saving..."):
                    msgs = process_employee(
                        emp_code=emp_code,
                        full_name=full_name,
                        department=department,
                        designation=designation,
                        mobile=mobile,
                        notes=notes,
                        uploaded_files=uploaded_files
                    )
                    st.session_state.save_messages = msgs

                has_error = any("error" in m.lower() or "fail" in m.lower() for m in msgs)
                if has_error:
                    st.session_state.save_result = "error"
                    st.error("Save had issues — read messages below")
                else:
                    st.session_state.save_result = "success"
                    st.success(f"**{emp_code}** saved/updated! Press Refresh in dashboard to see changes.")
                    st.session_state.last_processed = emp_code

                # Show messages persistently
                for m in st.session_state.save_messages:
                    if "error" in m.lower() or "fail" in m.lower():
                        st.error(m)
                    else:
                        st.info(m)

    with col_btn2:
        if st.button("Clear Form", use_container_width=True):
            keys = ["emp_code_input", "full_name", "department", "designation", "mobile", "notes", "uploader"]
            for k in keys:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.selected_emp_code = None
            st.session_state.last_processed = None
            st.session_state.save_result = None
            st.session_state.save_messages = []
            st.rerun()

# ────────────────────────────────────────────────
# Placeholder Pages (until attendance migrated)
# ────────────────────────────────────────────────
elif page == "Today's Attendance":
    st.subheader("Today's Attendance")
    st.info("Coming soon — attendance will be from Supabase.")

elif page == "Employee Attendance History":
    st.subheader("Employee Attendance History")
    st.info("Coming soon.")

elif page == "Daily Attendance Report":
    st.subheader("Daily Attendance Report")
    st.info("Coming soon.")
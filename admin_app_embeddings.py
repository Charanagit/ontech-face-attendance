import streamlit as st
import os
import io
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
import datetime
from supabase import create_client, Client
import base64

#################### SECTION 1: PAGE CONFIG & SUPABASE CLIENT ######################
st.set_page_config(page_title="Ontech Employee Manager", layout="wide")

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

try:
    supabase.table("employees").select("emp_code", count="planned").limit(0).execute()
    st.sidebar.success("Connected to Supabase ✓")
except Exception as e:
    st.sidebar.error(f"Supabase connection issue: {str(e)}")

#################### SECTION 2: STYLING & COLORS ######################
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

#################### SECTION 3: PATHS & SESSION STATE ######################
BASE_FOLDER = "data"
DATASET_FOLDER = os.path.join(BASE_FOLDER, "dataset")
os.makedirs(DATASET_FOLDER, exist_ok=True)

if "selected_emp_code" not in st.session_state:
    st.session_state.selected_emp_code = None
if "last_processed" not in st.session_state:
    st.session_state.last_processed = None
if "save_result" not in st.session_state:
    st.session_state.save_result = None
if "save_messages" not in st.session_state:
    st.session_state.save_messages = []

#################### SECTION 4: FACE MODEL (CACHED) ######################
@st.cache_resource
def get_face_model():
    with st.spinner("Loading InsightFace buffalo_s model..."):
        app = FaceAnalysis(name="buffalo_s")
        app.prepare(ctx_id=0, det_size=(640, 640))
    st.success("Model loaded", icon="✅")
    return app

app = get_face_model()

#################### SECTION 5: UTILITY FUNCTIONS ######################
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
    if isinstance(emb, str):  # in case base64
        return f"Yes (base64)"
    return "Yes"

#################### SECTION 6: CORE PROCESSING FUNCTION (UPDATED WITH BASE64 FIX) ######################
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

        with st.spinner("Processing face photos..."):
            for up_file in uploaded_files:
                try:
                    if up_file.size > 5 * 1024 * 1024:
                        messages.append(f"{up_file.name} too large (>5MB) → skipped")
                        continue

                    img_cv, img_pil = load_image(up_file.getvalue())
                    if img_cv is None:
                        continue

                    faces = app.get(img_cv)
                    if len(faces) != 1:
                        messages.append(f"{up_file.name}: {len(faces)} faces → skipped")
                        continue

                    face = faces[0]
                    if face.det_score < 0.75:
                        messages.append(f"{up_file.name}: low confidence → skipped")
                        continue

                    embeddings.append(normalize(face.embedding))
                    fname = os.path.splitext(up_file.name)[0] + ".png"
                    img_pil.save(os.path.join(emp_folder, fname))
                except Exception as e:
                    messages.append(f"Error processing {up_file.name}: {str(e)}")

        if len(embeddings) >= 3:
            mean_emb = normalize(np.mean(embeddings, axis=0))
            embedding_bytes = mean_emb.tobytes()
            expected_size = 512 * 4  # buffalo_s = 512-dim float32
            actual_size = len(embedding_bytes)
            if actual_size == expected_size:
                embedding_to_save = embedding_bytes
                messages.append(f"Embedding ready ({actual_size} bytes)")
            else:
                messages.append(f"Embedding size invalid ({actual_size} bytes, expected {expected_size}) → skipped")
        elif len(embeddings) > 0:
            messages.append(f"⚠️ Only {len(embeddings)} valid images (need ≥3)")
        else:
            messages.append("⚠️ No usable face images → saving without embedding")

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
                # Do NOT send registered_date → DB default now() will handle it
            }

            # BASE64 FIX HERE
            import base64
            if embedding_to_save:
                base64_encoded = base64.b64encode(embedding_to_save).decode('utf-8')
                data["embedding"] = base64_encoded
                messages.append(f"Embedding encoded as base64 ({len(base64_encoded)} chars)")

            response = supabase.table("employees").upsert(
                data,
                on_conflict="emp_code"
            ).execute()

            messages.append(f"Upsert affected {response.count} row(s)")
            if response.data:
                messages.append("Saved preview: " + str(response.data[0]))

            return messages

        except Exception as e:
            error_str = str(e)
            messages.append(f"Supabase error: {error_str}")
            st.error(f"Save failed: {error_str}")

            if "bytea" in error_str or "embedding" in error_str:
                st.warning("Embedding (bytea) issue — check if base64 encoding worked")
            if "permission" in error_str.lower() or "policy" in error_str.lower():
                st.warning("Permission / RLS problem — is RLS really disabled?")
            if "constraint" in error_str.lower():
                st.warning("Constraint violation (unique / not null / etc.)?")
            if "type" in error_str.lower():
                st.warning("Data type mismatch — check column types in Supabase")

            import traceback
            traceback.print_exc()
            return messages
        
#################### SECTION 7: MAIN UI & NAVIGATION ######################
st.title("🧑‍💼 Ontech Employee & Attendance Manager")
st.markdown(f"<h3 style='color:{PURPLE_ACCENT};'>Admin Control Panel</h3>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Section",
    ["Main Dashboard (Overview)", "Register / Edit Employee", "Today's Attendance", "Employee Attendance History", "Daily Attendance Report"]
)

#################### SECTION 8: MAIN DASHBOARD (OVERVIEW) ######################
if page == "Main Dashboard (Overview)":
    st.subheader("Registered Employees")

    try:
        # Fetch embedding presence too
        response = supabase.table("employees").select(
            "emp_code, full_name, department, designation, mobile, notes, embedding"
        ).execute()
        employees = response.data or []
    except Exception as e:
        st.error(f"Failed to load employees: {e}")
        employees = []

    if employees:
        # Prepare data with embedding status
        employees_data = []
        for emp in employees:
            employees_data.append({
                "Code": emp["emp_code"],
                "Name": emp.get("full_name") or emp["emp_code"],
                "Department": emp.get("department") or "",
                "Designation": emp.get("designation") or "",
                "Mobile": emp.get("mobile") or "",
                "Notes": emp.get("notes") or "",
                "Has Embedding": has_embedding(emp)
            })

        df = pd.DataFrame(employees_data)

        # Optional: color the embedding column
        def highlight_embedding(row):
            if row["Has Embedding"].startswith("Yes"):
                return ['background-color: #d4edda'] * len(row)  # light green
            else:
                return ['background-color: #f8d7da'] * len(row)  # light red

        styled_df = df.style.apply(highlight_embedding, axis=1)

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Has Embedding": st.column_config.TextColumn("Has Embedding", width="medium")
            }
        )

        if st.button("🔄 Refresh table", type="primary"):
            st.rerun()

    else:
        st.info("No employees registered yet. Add someone in 'Register / Edit Employee'.")
        
#################### SECTION 9: REGISTER / EDIT EMPLOYEE PAGE ######################
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
                full_name_value = emp.get("full_name", "")
                department_value = emp.get("department", "")
                designation_value = emp.get("designation", "")
                mobile_value = emp.get("mobile", "")
                notes_value = emp.get("notes", "")
                emp_code_value = code
                st.info(f"Editing employee: **{code}**", icon="✏️")
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

    st.subheader("Face Photos (3+ recommended, max 5MB each)")
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
                # Reset previous save state
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

                # Show messages persistently (no auto-rerun yet)
                has_error = any("error" in m.lower() or "fail" in m.lower() for m in msgs)

                if has_error:
                    st.session_state.save_result = "error"
                    st.error("Save had issues — read messages below")
                else:
                    st.session_state.save_result = "success"
                    st.success(f"**{emp_code}** saved successfully! Press Refresh table to see changes.")
                    st.session_state.last_processed = emp_code

                # Always show all messages after save attempt
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

#################### SECTION 10: OTHER PAGES (PLACEHOLDERS — STILL LOCAL) ######################
# Today's Attendance, History, Report — kept minimal / placeholder
# You can expand these later when migrating attendance to Supabase

elif page == "Today's Attendance":
    st.subheader("Today's Attendance")
    st.info("Attendance still using local DB — coming soon to Supabase.")

elif page == "Employee Attendance History":
    st.subheader("Employee Attendance History")
    st.info("Attendance history still local — coming soon.")

elif page == "Daily Attendance Report":
    st.subheader("Daily Attendance Report")
    st.info("Daily reports still local — coming soon.")
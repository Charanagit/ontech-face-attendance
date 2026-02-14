# admin_app_embeddings.py
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


st.set_page_config(page_title="Ontech Employee Manager", layout="wide")


# ────────────────────────────────────────────────
# Supabase client (using secrets from Streamlit Cloud)
# Make sure you added these in Settings → Secrets:
# SUPABASE_URL = "https://crujjurupavknjwdjjmj.supabase.co"
# SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
# ────────────────────────────────────────────────
supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

# Quick connection test in sidebar
try:
    supabase.table("employees").select("emp_code", count="planned").limit(0).execute()
    st.sidebar.success("Connected to Supabase ✓")
except Exception as e:
    st.sidebar.error(f"Supabase connection issue: {str(e)}")

# ────────────────────────────────────────────────
# Color Palette
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

# Page config & styling

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
# Paths (only for storing photos locally)
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

# ────────────────────────────────────────────────
# Model (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def get_face_model():
    with st.spinner("Initializing InsightFace model..."):
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


# ────────────────────────────────────────────────
# Core logic: process employee + photos → save to Supabase
# ────────────────────────────────────────────────
def process_employee(emp_code, full_name, department, designation, mobile, notes, uploaded_files):
    messages = []

    embedding_to_save = None

    if uploaded_files:
        embeddings = []
        emp_folder = os.path.join(DATASET_FOLDER, emp_code)
        os.makedirs(emp_folder, exist_ok=True)

        for up_file in uploaded_files:
            img_cv, img_pil = load_image(up_file.getvalue())
            if img_cv is None:
                continue

            faces = app.get(img_cv)
            if len(faces) != 1:
                st.warning(f"{up_file.name}: {len(faces)} faces detected → skipped")
                continue

            face = faces[0]
            if face.det_score < 0.75:
                st.warning(f"{up_file.name}: low confidence ({face.det_score:.2f}) → skipped")
                continue

            embeddings.append(normalize(face.embedding))
            fname = os.path.splitext(up_file.name)[0] + ".png"
            img_pil.save(os.path.join(emp_folder, fname))

        if len(embeddings) >= 3:
            embedding_to_save = normalize(np.mean(embeddings, axis=0)).tolist()  # list for jsonb
            messages.append(f"Embedding created successfully ({len(embeddings)} images)")
        elif len(embeddings) > 0:
            messages.append(f"⚠️ Only {len(embeddings)} valid images (need ≥3)")
        else:
            messages.append("⚠️ No usable face images → embedding not saved")

    # Save to Supabase (upsert)
    with st.spinner("Saving to cloud..."):
        try:
            data = {
                "emp_code": emp_code.strip().upper(),
                "full_name": full_name.strip() or None,
                "department": department.strip() or None,
                "designation": designation.strip() or None,
                "mobile": mobile.strip() or None,
                "notes": notes.strip() or None,
                "registered_date": datetime.datetime.now().isoformat(),
            }
            if embedding_to_save:
                data["embedding"] = embedding_to_save

            supabase.table("employees").upsert(data).execute()
            messages.append(f"Employee **{emp_code}** saved/updated in Supabase")
        except Exception as e:
            messages.append(f"Supabase save failed: {str(e)}")
            st.error(f"Failed to save to cloud: {str(e)}")

    return messages


# ────────────────────────────────────────────────
# UI - Sidebar Navigation
# ────────────────────────────────────────────────
st.title("🧑‍💼 Ontech Employee & Attendance Manager")
st.markdown(f"<h3 style='color:{PURPLE_ACCENT};'>Admin Control Panel</h3>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Section",
    ["Main Dashboard (Overview)",
     "Register / Edit Employee",
     "Today's Attendance",
     "Employee Attendance History",
     "Daily Attendance Report"]
)


# ────────────────────────────────────────────────
# Main Dashboard (Overview) - Loads from Supabase
# ────────────────────────────────────────────────
if page == "Main Dashboard (Overview)":
    st.subheader("Registered Employees & Today's Attendance Status")

    try:
        response = supabase.table("employees").select("emp_code, full_name, department, designation, mobile, notes").execute()
        employees = response.data
    except Exception as e:
        st.error(f"Failed to load employees from Supabase: {e}")
        employees = []

    if employees:
        # Attendance still uses local DB for now (we can migrate later)
        today_records = get_today_present()
        present_count = len(today_records)
        today_attendance_dict = {r["emp_code"]: r["checkin_time"] for r in today_records}

        st.caption(f"**Today ({datetime.date.today():%Y-%m-%d})**: {present_count} / {len(employees)} employees checked in")

        employees_data = []
        for emp in employees:
            code = emp["emp_code"]
            name = emp["full_name"] or code
            dept = emp["department"] or ""
            desig = emp["designation"] or ""
            mob = emp["mobile"] or ""
            notes = emp["notes"] or ""

            checkin_time = today_attendance_dict.get(code)
            present_str = f"Yes – {checkin_time}" if checkin_time else "No"

            employees_data.append({
                "Code": code,
                "Name": name,
                "Department": dept,
                "Designation": desig,
                "Mobile": mob,
                "Notes": notes[:100] + "…" if len(notes) > 100 else notes,
                "Present Today": present_str
            })

        df = pd.DataFrame(employees_data)

        def highlight_present(row):
            color = LIGHT_GREEN if row["Present Today"].startswith("Yes") else "#f8d7da"
            return [f'background-color: {color}' if col == "Present Today" else '' for col in df.columns]

        styled_df = df.style.apply(highlight_present, axis=1)

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Notes": st.column_config.TextColumn(width="medium"),
                "Present Today": st.column_config.TextColumn(width="small")
            }
        )

        if st.button("🔄 Refresh Attendance Status"):
            st.rerun()

        st.markdown("**Quick Edit Employee:**")
        cols = st.columns(6)
        for i, emp in enumerate(employees_data):
            with cols[i % 6]:
                label = f"✏ {emp['Code']}"
                if emp['Name'] and emp['Name'] != emp['Code']:
                    label += f" – {emp['Name'][:15]}…"
                if st.button(label, key=f"edit_{emp['Code']}", use_container_width=True):
                    st.session_state.selected_emp_code = emp["Code"]
                    st.rerun()
    else:
        st.info("No employees registered in Supabase yet. Add someone below.", icon="ℹ️")


# ────────────────────────────────────────────────
# Register / Edit Employee Page
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
                st.info(f"Editing employee: **{code}**", icon="✏️")
            else:
                st.warning("Employee not found in Supabase")
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

    st.subheader("Face Photos (3+ recommended)")
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
                cols[i % len(cols)].image(img, caption=file.name, use_column_width=True)
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
                if st.session_state.last_processed == emp_code:
                    st.info("Already processed this code in this session.")
                else:
                    with st.spinner("Processing employee..."):
                        msgs = process_employee(
                            emp_code=emp_code,
                            full_name=full_name,
                            department=department,
                            designation=designation,
                            mobile=mobile,
                            notes=notes,
                            uploaded_files=uploaded_files
                        )
                        for m in msgs:
                            st.info(m)
                        st.session_state.last_processed = emp_code
                    st.rerun()

    with col_btn2:
        if st.button("Clear Form", use_container_width=True):
            for k in ["emp_code_input", "full_name", "department", "designation", "mobile", "notes", "uploader"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.selected_emp_code = None
            st.rerun()


# ────────────────────────────────────────────────
# Today's Attendance Page (still using local DB for now)
# ────────────────────────────────────────────────
elif page == "Today's Attendance":
    st.subheader("Today's Attendance")

    today = datetime.date.today().isoformat()
    records = get_today_present()

    if records:
        df_today = pd.DataFrame(records)
        df_today["Status"] = df_today["checkout_time"].apply(lambda x: "Checked Out" if x != "-" else "Present")
        st.dataframe(
            df_today[["emp_code", "name", "department", "checkin_time", "checkout_time", "Status"]],
            use_container_width=True,
            hide_index=True
        )
        st.success(f"Total present today: {len(records)} employees")
    else:
        st.info("No attendance records today yet.", icon="ℹ️")


# ────────────────────────────────────────────────
# Employee Attendance History Page (local for now)
# ────────────────────────────────────────────────
elif page == "Employee Attendance History":
    st.subheader("Employee Attendance History")

    all_employees = get_all_employees()
    emp_options = [f"{code} - {name or code}" for code, name, _, _, _, _ in all_employees]
    emp_dict = {opt: code for opt, code in zip(emp_options, [r[0] for r in all_employees])}

    selected = st.selectbox("Select Employee", [""] + emp_options)

    if selected and selected != "":
        code = emp_dict[selected]
        name, _, _, _, _ = load_employee_info(code)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date", value=datetime.date.today() - datetime.timedelta(days=30))
        with col2:
            end_date = st.date_input("To Date", value=datetime.date.today())

        if st.button("Show History"):
            history = get_attendance_history_for_employee(code, limit=100)
            filtered = [r for r in history if start_date.isoformat() <= r["date"] <= end_date.isoformat()]

            if filtered:
                df_hist = pd.DataFrame(filtered)
                st.dataframe(
                    df_hist[["date", "checkin_time", "checkout_time", "status"]],
                    use_container_width=True,
                    hide_index=True
                )
                st.success(f"Found {len(filtered)} records between {start_date} and {end_date}")
            else:
                st.warning("No attendance records found in selected date range.")


# ────────────────────────────────────────────────
# Daily Attendance Report Page (local for now)
# ────────────────────────────────────────────────
elif page == "Daily Attendance Report":
    st.subheader("Daily Attendance Report")

    selected_date = st.date_input("Select Date", value=datetime.date.today())

    if st.button("View Report"):
        date_str = selected_date.isoformat()
        daily_records = get_attendance_for_date(date_str)

        if daily_records:
            df_daily = pd.DataFrame(daily_records)
            df_daily["Status"] = df_daily["checkout_time"].apply(lambda x: "Checked Out" if x != "-" else "Present")
            st.dataframe(
                df_daily[["emp_code", "name", "department", "checkin_time", "checkout_time", "Status"]],
                use_container_width=True,
                hide_index=True
            )
            st.success(f"Total records on {date_str}: {len(daily_records)}")
        else:
            st.info(f"No attendance records found for {date_str}.", icon="ℹ️")
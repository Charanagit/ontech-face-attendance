# admin_app_embeddings.py
import streamlit as st
import os
import io
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
import datetime

from database import (
    save_employee,
    load_employee_info,
    init_db,
    is_present_today,
    get_today_present,
    get_attendance_for_date,
    get_all_employees,
    get_attendance_history_for_employee,
    get_attendance_for_employee_date
)

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
st.set_page_config(page_title="Ontech Employee Manager", layout="wide")

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
# Config / Paths
# ────────────────────────────────────────────────
BASE_FOLDER = "data"
DATASET_FOLDER = os.path.join(BASE_FOLDER, "dataset")
EMBEDDINGS_FILE = os.path.join(BASE_FOLDER, "face_db.pkl")

os.makedirs(DATASET_FOLDER, exist_ok=True)
init_db()

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
    
def load_face_db():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_face_db(face_db):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(face_db, f)


def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


# ────────────────────────────────────────────────
# Core logic: process employee + photos
# ────────────────────────────────────────────────
def process_employee(emp_code, full_name, department, designation, mobile, notes, uploaded_files):
    messages = []

    save_employee(
        emp_code=emp_code,
        full_name=full_name,
        department=department,
        designation=designation,
        mobile=mobile,
        notes=notes
    )
    messages.append(f"Employee **{emp_code}** saved/updated")

    if uploaded_files:
        face_db = load_face_db()
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
            mean_emb = normalize(np.mean(embeddings, axis=0))
            face_db[emp_code] = mean_emb
            save_face_db(face_db)
            messages.append(f"Embedding created successfully ({len(embeddings)} images)")
        elif len(embeddings) > 0:
            messages.append(f"⚠️ Only {len(embeddings)} valid images (need ≥3)")
        else:
            messages.append("⚠️ No usable face images → embedding unchanged")

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
# Main Dashboard (Overview) - Restored + Improved
# ────────────────────────────────────────────────
if page == "Main Dashboard (Overview)":
    st.subheader("Registered Employees & Today's Attendance Status")

    face_db = load_face_db()

    if face_db:
        # Get today's attendance
        today_records = get_today_present()
        present_count = len(today_records)
        today_attendance_dict = {r["emp_code"]: r["checkin_time"] for r in today_records}

        st.caption(f"**Today ({datetime.date.today():%Y-%m-%d})**: {present_count} / {len(face_db)} employees checked in")

        employees_data = []
        for code in sorted(face_db.keys()):
            name, dept, desig, mob, notes = load_employee_info(code)
            checkin_time = today_attendance_dict.get(code)
            present_str = f"Yes – {checkin_time}" if checkin_time else "No"

            employees_data.append({
                "Code": code,
                "Name": name,
                "Department": dept,
                "Designation": desig,
                "Mobile": mob,
                "Notes": notes[:100] + "…" if len(notes or "") > 100 else (notes or ""),
                "Present Today": present_str,
                "Embedding": "Yes"
            })

        df = pd.DataFrame(employees_data)

        # Highlight present/absent rows
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

        # Refresh button
        if st.button("🔄 Refresh Attendance Status"):
            st.rerun()

        # Quick edit buttons
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
        st.info("No employees registered yet. Register someone using the form below.", icon="ℹ️")


# ────────────────────────────────────────────────
# Register / Edit Employee Page
# ────────────────────────────────────────────────
elif page == "Register / Edit Employee":
    st.subheader("Add / Edit Employee")

    emp_code_value = full_name_value = department_value = designation_value = mobile_value = notes_value = ""

    editing = bool(st.session_state.selected_emp_code)

    if editing:
        code = st.session_state.selected_emp_code
        name, dept, desig, mob, nt = load_employee_info(code)
        
        full_name_value = name if name and name != code else ""
        department_value = dept or ""
        designation_value = desig or ""
        mobile_value = mob or ""
        notes_value = nt or ""
        
        emp_code_value = code
        
        st.info(f"Editing employee: **{code}**", icon="✏️")

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
                else emp_code_input.strip()
            )

            if not emp_code:
                st.error("Employee Code is required")
            else:
                if st.session_state.last_processed == emp_code:
                    st.info("Already processed this code in this session.")
                else:
                    with st.spinner("Processing..."):
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
# Today's Attendance Page
# ────────────────────────────────────────────────
elif page == "Today's Attendance":
    st.subheader("Today's Attendance")

    today = datetime.date.today().isoformat()
    records = get_attendance_for_date(today)

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
# Employee Attendance History Page
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
# Daily Attendance Report Page
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
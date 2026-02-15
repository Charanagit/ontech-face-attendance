import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import numpy as np
import os
import time
import sys
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import winsound
import base64
import mediapipe as mp

# Supabase
SUPABASE_URL = "https://crujjurupavknjwdjjmj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNydWpqdXJ1cGF2a25qd2Rqam1qIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA5NjI0MTAsImV4cCI6MjA4NjUzODQxMH0.MdQDrEHOyQ0mI6HGX986lNMw5cpj5pfUCnKFh88pnzw"

from supabase import create_client, Client

# ────────────────────────────────────────────────
# Supabase Initialization
# ────────────────────────────────────────────────
supabase = None
supabase_connected = False
last_sync_time = None

print("=== Supabase Init ===")
try:
    supabase = create_client(SUPABASE_URL.strip(), SUPABASE_KEY.strip())
    # Test connection
    supabase.table("employees").select("emp_code", count="planned").limit(0).execute()
    supabase_connected = True
    print("Supabase connected successfully")
except Exception as e:
    print(f"Supabase connection failed: {str(e)}")
    import traceback
    traceback.print_exc()
    messagebox.showerror("Cloud Error", f"Supabase failed:\n{str(e)}\nApp will exit.")
    sys.exit(1)

# ────────────────────────────────────────────────
# Paths & Config
# ────────────────────────────────────────────────
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

COLOR_SUCCESS = (0, 255, 120)
COLOR_WARNING = (0, 80, 255)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_CLOUD_OK = (0, 200, 100)
COLOR_CLOUD_FAIL = (0, 80, 255)

SIMILARITY_THRESHOLD    = 0.37
GESTURE_HOLD_SECONDS    = 2.8
MIN_TIME_BETWEEN_ACTIONS = 5.0
SUCCESS_SHOW_SECONDS    = 5.0
PROCESS_EVERY_N_FRAMES  = 4

# ────────────────────────────────────────────────
# Globals
# ────────────────────────────────────────────────
face_db = {}  # emp_code → embedding (np.array)
employee_info = {}  # emp_code → {"full_name", "department", ...}
success_message_start = None
success_message_text = ""
gesture_active_until = 0.0
last_action_time = {}

# ────────────────────────────────────────────────
# Lazy-load InsightFace
# ────────────────────────────────────────────────
face_analyzer = None

def get_face_analyzer():
    global face_analyzer
    if face_analyzer is None:
        try:
            from insightface.app import FaceAnalysis
            print("Loading InsightFace buffalo_s ...")
            face_analyzer = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
            face_analyzer.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.32)
            print("InsightFace ready")
        except Exception as e:
            print(f"InsightFace failed: {e}")
            face_analyzer = None
    return face_analyzer

# ────────────────────────────────────────────────
# Load all data from Supabase
# ────────────────────────────────────────────────
def load_all_from_supabase():
    global face_db, employee_info, last_sync_time
    face_db = {}
    employee_info = {}

    if not supabase_connected:
        messagebox.showerror("No Cloud", "Supabase not connected.")
        return False

    try:
        response = supabase.table("employees").select(
            "emp_code, full_name, department, designation, mobile, notes, embedding"
        ).execute()

        count = 0
        for row in response.data:
            code = row["emp_code"]
            employee_info[code] = {
                "full_name": row.get("full_name", code),
                "department": row.get("department", ""),
                "designation": row.get("designation", ""),
                "mobile": row.get("mobile", ""),
                "notes": row.get("notes") or "",
            }

            emb_raw = row.get("embedding")
            if emb_raw is not None:
                        try:
                            print(f"Raw embedding type: {type(emb_raw).__name__}, preview: {str(emb_raw)[:50]}...")

                            if isinstance(emb_raw, bytes):
                                emb_bytes = emb_raw
                                print(f"Direct bytes received → length {len(emb_bytes)}")

                            elif isinstance(emb_raw, str) and emb_raw.startswith("\\x"):
                                # Postgres hex dump (most common case now)
                                hex_str = emb_raw[2:]  # remove \x prefix
                                emb_bytes = bytes.fromhex(hex_str)
                                print(f"Converted hex dump → {len(emb_bytes)} bytes")

                            elif isinstance(emb_raw, str):
                                # Fallback: assume base64
                                emb_clean = emb_raw.strip().replace("\n", "").replace(" ", "")
                                padding = (4 - len(emb_clean) % 4) % 4
                                emb_clean += "=" * padding
                                emb_bytes = base64.b64decode(emb_clean, validate=False)
                                print(f"Base64 fallback → {len(emb_bytes)} bytes")

                            else:
                                print("Unknown embedding format → skipping")
                                continue

                            # Convert to float32 array
                            emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
                            actual_len = len(emb_array)
                            print(f"Array length: {actual_len} floats")

                            # Accept any reasonable size for now
                            if 400 <= actual_len <= 800:
                                face_db[code] = emb_array
                                count += 1
                                print(f"→ LOADED {code} ({actual_len} floats)")
                            else:
                                print(f"→ REJECTED {code}: {actual_len} floats (unusual size)")

                        except Exception as e:
                            print(f"→ PARSE FAILED {code}: {str(e)}")
                            continue
        
        last_sync_time = datetime.datetime.now()
        print(f"\nFinal: {len(employee_info)} employees, {count} embeddings loaded")
        if count == 0 and len(employee_info) > 0:
            print("WARNING: No valid embeddings parsed")
        return True

    except Exception as e:
        print(f"Sync failed: {e}")
        messagebox.showerror("Sync Error", str(e))
        return False# ────────────────────────────────────────────────
def mark_present(emp_code: str) -> bool:
    emp_code = emp_code.strip().upper()
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")

    try:
        # Check if already checked in today
        existing = supabase.table("attendance")\
            .select("id")\
            .eq("emp_code", emp_code)\
            .eq("checkin_date", today)\
            .execute()

        if existing.data:
            print(f"{emp_code} already checked in today")
            return False

        supabase.table("attendance").insert({
            "emp_code": emp_code,
            "checkin_date": today,
            "checkin_time": now_time
        }).execute()

        print(f"Check-in recorded: {emp_code}")
        return True
    except Exception as e:
        print(f"Check-in failed: {e}")
        return False

def mark_out(emp_code: str) -> bool:
    emp_code = emp_code.strip().upper()
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")

    try:
        # Find latest unchecked-out record today
        response = supabase.table("attendance")\
            .select("id, checkout_time")\
            .eq("emp_code", emp_code)\
            .eq("checkin_date", today)\
            .order("id", desc=True)\
            .limit(1)\
            .execute()

        if not response.data or response.data[0]["checkout_time"]:
            print(f"No open check-in for {emp_code} today")
            return False

        record_id = response.data[0]["id"]

        supabase.table("attendance")\
            .update({"checkout_time": now_time})\
            .eq("id", record_id)\
            .execute()

        print(f"Check-out recorded: {emp_code}")
        return True
    except Exception as e:
        print(f"Check-out failed: {e}")
        return False

# ────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def is_victory_gesture(lm):
    if not lm:
        return False
    return (
        lm[8].y  < lm[6].y  - 0.018 and
        lm[12].y < lm[10].y - 0.018 and
        lm[16].y > lm[14].y + 0.008 and
        lm[20].y > lm[18].y + 0.008
    )

# ────────────────────────────────────────────────
# Splash (unchanged)
# ────────────────────────────────────────────────
def show_splash():
    logo_path = os.path.join(BASE_DIR, "OnTech.png")
    if not os.path.exists(logo_path):
        print("Logo missing — skipping")
        return True

    logo = cv2.imread(logo_path)
    if logo is None:
        print("Logo load failed")
        return True

    h, w = logo.shape[:2]
    win_name = "Ontech Face Recognition"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    bg = np.zeros((720, 1280, 3), dtype=np.uint8)
    x_offset = (1280 - w) // 2
    y_offset = (720 - h) // 2
    bg[y_offset:y_offset+h, x_offset:x_offset+w] = logo

    for alpha in range(0, 256, 6):
        blended = cv2.addWeighted(bg, alpha/255.0, np.zeros_like(bg), 1 - alpha/255.0, 0)
        cv2.imshow(win_name, blended)
        cv2.waitKey(25)

    cv2.waitKey(1500)

    for alpha in range(255, -1, 8):
        blended = cv2.addWeighted(bg, alpha/255.0, np.zeros_like(bg), 1 - alpha/255.0, 0)
        cv2.imshow(win_name, blended)
        cv2.waitKey(20)

    cv2.destroyAllWindows()
    return True

# ────────────────────────────────────────────────
# UI Windows (updated for Supabase)
# ────────────────────────────────────────────────
def show_employee_list():
    top = tk.Toplevel()
    top.title("Registered Employees (Cloud)")
    top.geometry("1100x700")
    top.minsize(1000, 600)

    tree = ttk.Treeview(top, columns=("Code","Name","Dept","Desig","Mobile","Notes"), show="headings")
    tree.heading("Code", text="Code")
    tree.heading("Name", text="Name")
    tree.heading("Dept", text="Department")
    tree.heading("Desig", text="Designation")
    tree.heading("Mobile", text="Mobile")
    tree.heading("Notes", text="Notes")

    tree.column("Code", width=90, anchor="center")
    tree.column("Name", width=200)
    tree.column("Dept", width=140)
    tree.column("Desig", width=160)
    tree.column("Mobile", width=120)
    tree.column("Notes", width=250)

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    sb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
    tree.configure(yscroll=sb.set)
    sb.pack(side="right", fill="y")

    for code, info in employee_info.items():
        # Safe handling for notes (could be None or empty)
        notes = info.get("notes") or ""   # default to empty string if None
        display_notes = notes[:90] + "…" if len(notes) > 90 else notes

        tree.insert("", "end", values=(
            code,
            info["full_name"],
            info["department"],
            info["designation"],
            info["mobile"],
            display_notes
        ))

    if not employee_info:
        tree.insert("", "end", values=("", "No employees loaded from cloud", "", "", "", ""))
def show_today_attendance():
    top = tk.Toplevel()
    top.title("Today's Attendance (Cloud)")
    top.geometry("900x600")

    tree = ttk.Treeview(top, columns=("Code","Name","Dept","Check-in","Check-out"), show="headings")
    tree.heading("Code", text="Code")
    tree.heading("Name", text="Name")
    tree.heading("Dept", text="Department")
    tree.heading("Check-in", text="Check-in")
    tree.heading("Check-out", text="Check-out")

    tree.column("Code", width=100, anchor="center")
    tree.column("Name", width=220)
    tree.column("Dept", width=180)
    tree.column("Check-in", width=140)
    tree.column("Check-out", width=140)

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    sb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
    tree.configure(yscroll=sb.set)
    sb.pack(side="right", fill="y")

    try:
        today = datetime.date.today().isoformat()
        records = supabase.table("attendance")\
            .select("emp_code, checkin_time, checkout_time")\
            .eq("checkin_date", today)\
            .execute().data

        for r in records:
            code = r["emp_code"]
            name = employee_info.get(code, {}).get("full_name", code)
            dept = employee_info.get(code, {}).get("department", "")
            tree.insert("", "end", values=(
                code, name, dept,
                r["checkin_time"] or "-",
                r["checkout_time"] or "-"
            ))

        if not records:
            tree.insert("", "end", values=("", "No attendance today", "", "", ""))
    except Exception as e:
        tree.insert("", "end", values=("", f"Error loading attendance: {str(e)}", "", "", ""))

# ────────────────────────────────────────────────
# Recognition Loop
# ────────────────────────────────────────────────
def run_attendance_recognition():
    global success_message_start, success_message_text, gesture_active_until

    analyzer = get_face_analyzer()
    if analyzer is None:
        messagebox.showerror("Error", "Cannot load face model.")
        return

    # Load from cloud
    if not load_all_from_supabase():
        messagebox.showwarning("Warning", "Failed to load faces from cloud.\nOnly 'Unknown' will be detected.")

    if not face_db:
        messagebox.showwarning("No Faces", "No registered faces with embeddings found.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WINDOW_NAME = "Ontech Attendance"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.52,
        min_tracking_confidence=0.52
    )

    frame_count = 0
    last_results = []
    prev_time = time.time()

    print("Started → Face = Check-in | Face + ✌️ hold ~3s = Check-out")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame lost → retrying...")
            cap.release()
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            time.sleep(0.5)
            continue

        display_frame = frame.copy()
        frame_count += 1
        now = time.time()

        # Gesture
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)
            if hand_results.multi_hand_landmarks:
                for hlm in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    if is_victory_gesture(hlm.landmark):
                        gesture_active_until = now + GESTURE_HOLD_SECONDS
                        winsound.Beep(1800, 80)
        except Exception as e:
            print(f"Hand error: {e}")

        # Face
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            try:
                faces = analyzer.get(frame)
                results = []

                for face in faces:
                    if face.det_score < 0.30:
                        continue
                    emb = normalize(face.embedding)
                    best_code = "Unknown"
                    best_score = -1.0

                    for code, db_emb in face_db.items():
                        sc = cosine_similarity(emb, db_emb)
                        if sc > best_score:
                            best_score = sc
                            best_code = code

                    info = employee_info.get(best_code, {"full_name": best_code, "department": ""})
                    display_name = f"{info['full_name']} ({info['department']})" if info['department'] else info['full_name']
                    bbox = face.bbox.astype(int)
                    results.append((bbox, display_name, best_score, face.det_score, best_code))

                last_results = results
            except Exception as e:
                print(f"Face error: {e}")
                last_results = []

        # Draw + action
        for bbox, dname, score, det_conf, code in last_results:
            x1,y1,x2,y2 = map(int, bbox)
            color = COLOR_SUCCESS if code != "Unknown" else COLOR_UNKNOWN

            cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 3)

            label = f"{dname} {score:.3f}"
            if code == "Unknown":
                label += f" det:{det_conf:.2f}"
            else:
                label += f" sim:{score:.3f}"

            tw = len(label) * 11 + 20
            cv2.rectangle(display_frame, (x1, y1-35), (x1 + tw, y1-5), color, -1)
            cv2.putText(display_frame, label, (x1+8, y1-12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0,0,0), 2)

            if code != "Unknown" and score >= SIMILARITY_THRESHOLD:
                if code in last_action_time and now - last_action_time[code] < MIN_TIME_BETWEEN_ACTIONS:
                    continue

                is_checkout = now < gesture_active_until
                success = False
                action_text = ""

                if is_checkout:
                    success = mark_out(code)
                    action_text = "CHECKED OUT"
                else:
                    success = mark_present(code)
                    action_text = "CHECKED IN"

                if success:
                    last_action_time[code] = now
                    winsound.Beep(1200, 400)
                    success_message_text = f"{action_text} → {dname.split(' (')[0]}"
                    success_message_start = now
                    print(f"SUCCESS: {action_text} {code}")

        # Success overlay
        if success_message_start and now - success_message_start < SUCCESS_SHOW_SECONDS:
            cv2.putText(display_frame, success_message_text, (140, 160),
                        cv2.FONT_HERSHEY_DUPLEX, 2.2, COLOR_SUCCESS, 6)
            cv2.putText(display_frame, datetime.datetime.now().strftime("%H:%M:%S"),
                        (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLOR_SUCCESS, 3)

        # Status
        status_text = f"Loaded {len(face_db)} faces from cloud"
        status_color = COLOR_CLOUD_OK if supabase_connected else COLOR_CLOUD_FAIL
        if last_sync_time:
            ago = (datetime.datetime.now() - last_sync_time).seconds // 60
            status_text += f" (synced {ago} min ago)"

        cv2.putText(display_frame, status_text, (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        fps = 1 / (now - prev_time) if now > prev_time else 0
        prev_time = now
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 3)

        cv2.putText(display_frame,
                    "Face = Check-in | Face + ✌️ (hold ~3s) = Check-out | ESC to close",
                    (25, display_frame.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (220,220,255), 2)

        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

# ────────────────────────────────────────────────
# Kiosk Dashboard
# ────────────────────────────────────────────────
def launch_kiosk():
    if not load_all_from_supabase():
        messagebox.showwarning("Cloud Warning", "Failed to load data from Supabase.\nSome features may be limited.")

    root = tk.Tk()
    root.title("Ontech Attendance Kiosk")
    root.geometry("720x680")
    root.configure(bg="#0f172a")
    root.resizable(False, False)

    header_frame = tk.Frame(root, bg="#0f172a")
    header_frame.pack(fill="x", pady=(30, 10))

    tk.Label(header_frame, text="Ontech Attendance",
             font=("Helvetica", 32, "bold"), fg="#f97316", bg="#0f172a").pack()

    sync_text = f"{datetime.date.today():%Y-%m-%d} • {len(face_db)} employees"
    if last_sync_time:
        ago = (datetime.datetime.now() - last_sync_time).seconds // 60
        sync_text += f" (cloud sync {ago} min ago)"

    tk.Label(header_frame, text=sync_text,
             font=("Helvetica", 14), fg="#94a3b8", bg="#0f172a").pack(pady=(4, 0))

    status_frame = tk.Frame(root, bg="#1e293b", bd=1, relief="flat")
    status_frame.pack(pady=20, padx=40, fill="x")

    status_color = "green" if supabase_connected else "red"
    tk.Label(status_frame, text="Ready to scan • Place face in front of camera",
             font=("Helvetica", 13), fg="#cbd5e1", bg="#1e293b", pady=12).pack()

    btn_frame = tk.Frame(root, bg="#0f172a")
    btn_frame.pack(pady=30, padx=60, fill="x")

    button_style = {
        "font": ("Helvetica", 15, "bold"),
        "width": 28,
        "height": 2,
        "bd": 0,
        "relief": "flat",
        "cursor": "hand2",
        "activebackground": "#334155",
    }

    def create_button(text, command, bg, fg="#ffffff"):
        btn = tk.Button(btn_frame, text=text, command=command, bg=bg, fg=fg, **button_style)
        btn.pack(pady=14, fill="x")
        btn.bind("<Enter>", lambda e: btn.config(bg="#334155"))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    create_button(
        "Start Attendance Recognition",
        lambda: threading.Thread(target=run_attendance_recognition, daemon=True).start(),
        "#f97316", "#000000"
    )

    create_button("View All Registered Employees", show_employee_list, "#22c55e")
    create_button("Today's Attendance Records", show_today_attendance, "#06b6d4")

    create_button("Sync Faces from Cloud Now",
                  lambda: load_all_from_supabase() or messagebox.showinfo("Sync", f"Reloaded {len(face_db)} faces"),
                  "#8b5cf6", "#ffffff")

    create_button("Exit Application", root.quit, "#ef4444")

    footer = tk.Label(root, text="Ontech • Powered by InsightFace + MediaPipe + Supabase",
                      font=("Helvetica", 10), fg="#475569", bg="#0f172a")
    footer.pack(side="bottom", pady=20)

    root.mainloop()

# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────
if __name__ == "__main__":
    if show_splash():
        launch_kiosk()
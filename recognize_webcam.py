# recognize_webcam.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import numpy as np
import pickle
import os
import time
import sys
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import winsound

import mediapipe as mp

from database import (
    init_db,
    load_employee_info,
    mark_present,
    mark_out,
    is_present_today,
    get_today_present,
)

# ────────────────────────────────────────────────
# Paths & Config
# ────────────────────────────────────────────────
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

EMBEDDINGS_FILE = os.path.join(DATA_DIR, "face_db.pkl")
DB_PATH = os.path.join(DATA_DIR, "employees.db")

COLOR_SUCCESS = (0, 255, 120)
COLOR_WARNING = (0, 80, 255)
COLOR_UNKNOWN = (0, 0, 255)

# Tunable parameters
SIMILARITY_THRESHOLD    = 0.37      # lowered a bit - tune between 0.35–0.42
GESTURE_HOLD_SECONDS    = 2.8
MIN_TIME_BETWEEN_ACTIONS = 5.0      # seconds
SUCCESS_SHOW_SECONDS    = 5.0
PROCESS_EVERY_N_FRAMES  = 4

# ────────────────────────────────────────────────
# Globals
# ────────────────────────────────────────────────
face_db = {}
if os.path.exists(EMBEDDINGS_FILE):
    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            face_db = pickle.load(f)
        print(f"Loaded {len(face_db)} identities")
    except Exception as e:
        print(f"face_db load failed: {e}")

success_message_start = None
success_message_text = ""
gesture_active_until = 0.0          # main fix for checkout persistence
last_action_time = {}               # per employee cooldown

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
            print(f"InsightFace init failed: {e}")
            face_analyzer = None
    return face_analyzer

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
    # Slightly more forgiving thresholds
    return (
        lm[8].y  < lm[6].y  - 0.018 and   # index tip above PIP
        lm[12].y < lm[10].y - 0.018 and   # middle tip above PIP
        lm[16].y > lm[14].y + 0.008 and   # ring down
        lm[20].y > lm[18].y + 0.008       # pinky down
    )

# ────────────────────────────────────────────────
# Splash screen (kept as is)
# ────────────────────────────────────────────────
def show_splash():
    logo_path = os.path.join(BASE_DIR, "OnTech.png")
    if not os.path.exists(logo_path):
        print("Logo not found — skipping splash")
        return True

    logo = cv2.imread(logo_path)
    if logo is None:
        print("Failed to load logo")
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

    for alpha in range(255, -1, -8):
        blended = cv2.addWeighted(bg, alpha/255.0, np.zeros_like(bg), 1 - alpha/255.0, 0)
        cv2.imshow(win_name, blended)
        cv2.waitKey(20)

    cv2.destroyAllWindows()
    return True

# ────────────────────────────────────────────────
# UI helper windows (kept almost as is)
# ────────────────────────────────────────────────
def show_employee_list():
    top = tk.Toplevel()
    top.title("Registered Employees")
    top.geometry("1100x700")
    top.minsize(1000, 600)

    tree = ttk.Treeview(top, columns=("Code","Name","Dept","Desig","Mobile","Notes","Today"), show="headings")
    tree.heading("Code", text="Code")
    tree.heading("Name", text="Name")
    tree.heading("Dept", text="Department")
    tree.heading("Desig", text="Designation")
    tree.heading("Mobile", text="Mobile")
    tree.heading("Notes", text="Notes")
    tree.heading("Today", text="Present Today")

    # column widths...
    tree.column("Code", width=90, anchor="center")
    tree.column("Name", width=170)
    tree.column("Dept", width=130)
    tree.column("Desig", width=150)
    tree.column("Mobile", width=110)
    tree.column("Notes", width=220)
    tree.column("Today", width=140, anchor="center")

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    sb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
    tree.configure(yscroll=sb.set)
    sb.pack(side="right", fill="y")

    if not face_db:
        tree.insert("", "end", values=("", "No employees yet", "", "", "", "", ""))
        return

    for code in sorted(face_db):
        name, dept, desig, mob, notes = load_employee_info(code)
        pres = is_present_today(code)
        tag = "present" if pres.startswith("Yes") else "absent"
        tree.insert("", "end", values=(code, name, dept, desig, mob,
                                       notes[:90]+"…" if len(notes or "")>90 else (notes or ""),
                                       pres), tags=(tag,))

    tree.tag_configure("present", foreground="green")
    tree.tag_configure("absent", foreground="gray")

def show_today_attendance():
    top = tk.Toplevel()
    top.title("Today's Check-ins")
    top.geometry("900x600")

    tree = ttk.Treeview(top, columns=("Code","Name","Dept","Check-in","Check-out"), show="headings")
    tree.heading("Code", text="Code")
    tree.heading("Name", text="Name")
    tree.heading("Dept", text="Department")
    tree.heading("Check-in", text="Check-in Time")
    tree.heading("Check-out", text="Check-out Time")

    tree.column("Code", width=100, anchor="center")
    tree.column("Name", width=220)
    tree.column("Dept", width=180)
    tree.column("Check-in", width=140, anchor="center")
    tree.column("Check-out", width=140, anchor="center")

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    sb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
    tree.configure(yscroll=sb.set)
    sb.pack(side="right", fill="y")

    records = get_today_present()
    for r in records:
        tree.insert("", "end", values=(
            r["emp_code"], r["name"], r["department"],
            r["checkin_time"], r["checkout_time"] if r["checkout_time"] != "-" else "-"
        ))

    if not records:
        tree.insert("", "end", values=("", "No check-ins today", "", "", ""))

# ────────────────────────────────────────────────
# Recognition loop (improved stability)
# ────────────────────────────────────────────────
def run_attendance_recognition():
    global success_message_start, success_message_text, gesture_active_until

    analyzer = get_face_analyzer()
    if analyzer is None:
        messagebox.showerror("Error", "Cannot load face model.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open webcam.")
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
        min_tracking_confidence=0.52,
    )

    frame_count = 0
    last_results = []
    prev_time = time.time()

    print("Recognition started → Face = Check-in | Face + ✌️ (hold ~3s) = Check-out")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame lost → trying to recover...")
            cap.release()
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            time.sleep(0.5)
            continue

        display_frame = frame.copy()
        frame_count += 1
        now = time.time()

        # ── Gesture detection every frame ──
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
                        winsound.Beep(1800, 80)  # short high beep = gesture seen
        except Exception as e:
            print(f"Hand processing error: {e}")

        # ── Face processing every N frames ──
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

                    name, dept, *_ = load_employee_info(best_code)
                    display_name = f"{name} ({dept})" if dept else name
                    bbox = face.bbox.astype(int)
                    results.append((bbox, display_name, best_score, face.det_score, best_code))

                last_results = results
            except Exception as e:
                print(f"Face processing error: {e}")
                last_results = []

        # ── Draw results & decide action ──
        for bbox, dname, score, det_conf, code in last_results:
            x1,y1,x2,y2 = map(int, bbox)
            color = COLOR_SUCCESS if dname != "Unknown" else COLOR_UNKNOWN

            cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 3)

            label = f"{dname} {score:.3f}"
            if dname == "Unknown":
                label += f"  det:{det_conf:.2f}"
            else:
                label += f"  sim:{score:.3f}"

            tw = len(label) * 11 + 20
            cv2.rectangle(display_frame, (x1, y1-35), (x1 + tw, y1-5), color, -1)
            cv2.putText(display_frame, label, (x1+8, y1-12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0,0,0), 2)

            if code != "Unknown" and score >= SIMILARITY_THRESHOLD:
                if code in last_action_time and now - last_action_time[code] < MIN_TIME_BETWEEN_ACTIONS:
                    continue

                is_checkout = (now < gesture_active_until)

                success = False
                action_text = ""

                if is_checkout:
                    print(f"CHECK-OUT attempt → {code}  sim={score:.3f}")
                    success = mark_out(code)
                    action_text = "CHECKED OUT"
                else:
                    print(f"CHECK-IN attempt → {code}  sim={score:.3f}")
                    success = mark_present(code)
                    action_text = "CHECKED IN"

                if success:
                    last_action_time[code] = now
                    winsound.Beep(1200, 400)
                    success_message_text = f"{action_text} → {dname.split(' (')[0]}"
                    success_message_start = now
                    print(f"SUCCESS: {action_text} {code}")

        # ── Success overlay ──
        if success_message_start and (now - success_message_start < SUCCESS_SHOW_SECONDS):
            cv2.putText(display_frame, success_message_text, (140, 160),
                        cv2.FONT_HERSHEY_DUPLEX, 2.2, COLOR_SUCCESS, 6)
            cv2.putText(display_frame, datetime.datetime.now().strftime("%H:%M:%S"),
                        (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLOR_SUCCESS, 3)

        # ── FPS + hint ──
        fps = 1 / (now - prev_time) if now > prev_time else 0
        prev_time = now
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 3)

        cv2.putText(display_frame,
                    "Face = Check-in    |    Face + ✌️ (hold ~3s) = Check-out    |    ESC to close",
                    (25, display_frame.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (220,220,255), 2)

        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

# ────────────────────────────────────────────────
# Kiosk Dashboard (your original beautiful UI)
# ────────────────────────────────────────────────
def launch_kiosk():
    init_db()

    root = tk.Tk()
    root.title("Ontech Attendance Kiosk")
    root.geometry("720x680")
    root.configure(bg="#0f172a")
    root.resizable(False, False)

    header_frame = tk.Frame(root, bg="#0f172a")
    header_frame.pack(fill="x", pady=(30, 10))

    tk.Label(header_frame, text="Ontech Attendance",
             font=("Helvetica", 32, "bold"), fg="#f97316", bg="#0f172a").pack()

    tk.Label(header_frame,
             text=f"{datetime.date.today():%Y-%m-%d} • {len(face_db)} employees registered",
             font=("Helvetica", 14), fg="#94a3b8", bg="#0f172a").pack(pady=(4, 0))

    status_frame = tk.Frame(root, bg="#1e293b", bd=1, relief="flat")
    status_frame.pack(pady=20, padx=40, fill="x")

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
    create_button("Exit Application", root.quit, "#ef4444")

    footer = tk.Label(root, text="Ontech Face Recognition • Powered by InsightFace + MediaPipe",
                      font=("Helvetica", 10), fg="#475569", bg="#0f172a")
    footer.pack(side="bottom", pady=20)

    root.mainloop()

# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────
if __name__ == "__main__":
    if show_splash():
        launch_kiosk()
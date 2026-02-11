# database.py
import sqlite3
import os
import sys
import datetime
from typing import Tuple, List, Optional, Dict

# Consistent path (same as recognize_webcam.py)
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_FOLDER = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_FOLDER, "employees.db")


def normalize_emp_code(code: str) -> str:
    """Clean emp_code consistently — always uppercase for reliable matching."""
    if not code:
        return ""
    return str(code).strip().upper()


def init_db() -> None:
    """Initialize database and tables if they don't exist."""
    os.makedirs(BASE_FOLDER, exist_ok=True)
    
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        
        # Employees table
        c.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                emp_code        TEXT PRIMARY KEY,
                full_name       TEXT,
                department      TEXT,
                designation     TEXT,
                mobile          TEXT,
                registered_date TEXT,
                notes           TEXT
            )
        ''')
        
        # Attendance table
        c.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_code        TEXT NOT NULL,
                checkin_date    TEXT NOT NULL,
                checkin_time    TEXT NOT NULL,
                checkout_time   TEXT,
                FOREIGN KEY (emp_code) REFERENCES employees(emp_code)
            )
        ''')
        
        # Add checkout_time column if missing (migration safety)
        try:
            c.execute("ALTER TABLE attendance ADD COLUMN checkout_time TEXT")
            print("Added checkout_time column (if missing)")
        except sqlite3.OperationalError:
            pass  # already exists
        
        conn.commit()


def save_employee(
    emp_code: str,
    full_name: str = "",
    department: str = "",
    designation: str = "",
    mobile: str = "",
    notes: str = ""
) -> None:
    init_db()
    emp_code = normalize_emp_code(emp_code)
    if not emp_code:
        print("Error: emp_code cannot be empty")
        return
    
    registered_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO employees 
                (emp_code, full_name, department, designation, mobile, registered_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                emp_code,
                full_name.strip(),
                department.strip(),
                designation.strip(),
                mobile.strip(),
                registered_date,
                notes.strip()
            ))
            conn.commit()
            print(f"Employee saved/updated: {emp_code}")
    except sqlite3.Error as e:
        print(f"Database error saving employee {emp_code}: {e}")


def load_employee_info(emp_code: str) -> Tuple[str, str, str, str, str]:
    init_db()
    emp_code = normalize_emp_code(emp_code)
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                SELECT full_name, department, designation, mobile, notes 
                FROM employees 
                WHERE emp_code = ?
            ''', (emp_code,))
            row = c.fetchone()
            if row:
                full_name, dept, desig, mob, nt = row
                return (
                    full_name or emp_code,
                    dept or "",
                    desig or "",
                    mob or "",
                    nt or ""
                )
            return emp_code, "", "", "", ""
    except sqlite3.Error as e:
        print(f"Error loading employee {emp_code}: {e}")
        return emp_code, "", "", "", ""


def get_all_employees() -> List[Tuple[str, str, str, str, str, str]]:
    """Return list of all registered employees for debugging / admin."""
    init_db()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                SELECT emp_code, full_name, department, designation, mobile, notes 
                FROM employees 
                ORDER BY emp_code
            ''')
            rows = c.fetchall()
            print(f"Total employees in database: {len(rows)}")
            return rows
    except sqlite3.Error as e:
        print(f"Error fetching all employees: {e}")
        return []


def mark_present(emp_code: str) -> bool:
    emp_code = normalize_emp_code(emp_code)
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            print(f"[mark_present] called for {emp_code} on {today}")
            
            c.execute('''
                SELECT 1 FROM attendance 
                WHERE emp_code = ? AND checkin_date = ?
            ''', (emp_code, today))
            
            if c.fetchone():
                print(f"[mark_present] {emp_code} already checked in today")
                return False
            
            c.execute('''
                INSERT INTO attendance (emp_code, checkin_date, checkin_time)
                VALUES (?, ?, ?)
            ''', (emp_code, today, now_time))
            conn.commit()
            print(f"[mark_present] Check-in recorded → {emp_code} at {now_time}")
            return True
    except sqlite3.Error as e:
        print(f"[mark_present] Check-in error for {emp_code}: {e}")
        return False


def mark_out(emp_code: str) -> bool:
    emp_code = normalize_emp_code(emp_code)
    today = datetime.date.today().isoformat()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            print(f"[mark_out] called for {emp_code} on {today}")
            
            c.execute("""
                SELECT checkin_time, checkout_time, id 
                FROM attendance 
                WHERE emp_code = ? AND checkin_date = ?
                ORDER BY id DESC LIMIT 1
            """, (emp_code, today))
            
            row = c.fetchone()
            print(f"[mark_out] Query result for {emp_code}: {row}")
            
            if not row or row[0] is None:
                print(f"[mark_out] No check-in found for {emp_code} today → cannot checkout")
                return False
            
            if row[1] is not None:
                print(f"[mark_out] {emp_code} already checked out at {row[1]} → skipping")
                return False
            
            record_id = row[2]
            c.execute("""
                UPDATE attendance 
                SET checkout_time = ? 
                WHERE id = ?
            """, (now_time, record_id))
            
            conn.commit()
            success = c.rowcount > 0
            print(f"[mark_out] Update rows affected: {c.rowcount}")
            if success:
                print(f"[mark_out] Check-out recorded → {emp_code} at {now_time}")
            else:
                print("[mark_out] Update failed - no rows matched")
            return success
    except sqlite3.Error as e:
        print(f"[mark_out] Check-out error for {emp_code}: {e}")
        return False


def is_present_today(emp_code: str) -> str:
    emp_code = normalize_emp_code(emp_code)
    today = datetime.date.today().isoformat()
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT checkin_time, checkout_time FROM attendance 
                WHERE emp_code = ? AND checkin_date = ?
                ORDER BY checkin_time DESC LIMIT 1
            """, (emp_code, today))
            row = c.fetchone()
            if row:
                checkin, checkout = row
                if checkout:
                    return f"Checked out at {checkout}"
                return f"Yes ({checkin})"
            return "No"
    except sqlite3.Error as e:
        print(f"[is_present_today] Presence check error for {emp_code}: {e}")
        return "Error"


def get_attendance_for_employee_date(emp_code: str, date: str) -> Dict[str, str]:
    """
    Check if employee was present on a specific date.
    Returns dict with 'status', 'checkin_time', 'checkout_time' or empty dict if absent.
    Date format: 'YYYY-MM-DD'
    """
    emp_code = normalize_emp_code(emp_code)
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT checkin_time, checkout_time 
                FROM attendance 
                WHERE emp_code = ? AND checkin_date = ?
                ORDER BY id DESC LIMIT 1
            """, (emp_code, date))
            
            row = c.fetchone()
            if row:
                checkin, checkout = row
                return {
                    "status": "present",
                    "checkin_time": checkin,
                    "checkout_time": checkout or "Not checked out"
                }
            return {"status": "absent"}
    except sqlite3.Error as e:
        print(f"[get_attendance_for_employee_date] Error for {emp_code} on {date}: {e}")
        return {"status": "error", "error": str(e)}


def get_attendance_history_for_employee(emp_code: str, limit: int = 30) -> List[Dict]:
    """
    Get recent attendance history for an employee (last N records)
    Returns list of dicts: date, checkin_time, checkout_time, status
    """
    emp_code = normalize_emp_code(emp_code)
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT checkin_date, checkin_time, checkout_time
                FROM attendance
                WHERE emp_code = ?
                ORDER BY checkin_date DESC
                LIMIT ?
            """, (emp_code, limit))
            
            rows = c.fetchall()
            history = []
            for date, cin, cout in rows:
                status = "Checked out" if cout else "Present"
                history.append({
                    "date": date,
                    "checkin_time": cin,
                    "checkout_time": cout or "-",
                    "status": status
                })
            return history
    except sqlite3.Error as e:
        print(f"[get_attendance_history_for_employee] Error for {emp_code}: {e}")
        return []


def get_attendance_for_date(date: str) -> List[Dict]:
    """
    Get all attendance records for a specific date (admin daily view)
    Returns list of dicts with employee info + attendance times
    Date format: 'YYYY-MM-DD'
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT 
                    e.emp_code, e.full_name, e.department,
                    a.checkin_time, a.checkout_time
                FROM attendance a
                JOIN employees e ON a.emp_code = e.emp_code
                WHERE a.checkin_date = ?
                ORDER BY a.checkin_time DESC
            """, (date,))
            
            rows = c.fetchall()
            return [
                {
                    "emp_code": code,
                    "name": name,
                    "department": dept,
                    "checkin_time": cin,
                    "checkout_time": cout or "-"
                }
                for code, name, dept, cin, cout in rows
            ]
    except sqlite3.Error as e:
        print(f"[get_attendance_for_date] Error for date {date}: {e}")
        return []


def get_today_present() -> List[Dict]:
    """Alias: returns today's attendance using current date"""
    today = datetime.date.today().isoformat()
    return get_attendance_for_date(today)


# ── Debug helper ──────────────────────────────────────────────
def debug_db_status():
    """Print quick overview of database contents."""
    print("\n=== Database Status Debug ====")
    employees = get_all_employees()
    print(f"Employees registered: {len(employees)}")
    if employees:
        print("First few:", employees[:3])
    
    today = datetime.date.today().isoformat()
    print(f"Attendance records today ({today}): {len(get_today_present())}")


if __name__ == "__main__":
    debug_db_status()
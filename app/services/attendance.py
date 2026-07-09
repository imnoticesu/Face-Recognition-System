"""
Attendance — ระบบบันทึกเวลาเข้า/ออกงานด้วยใบหน้า (SQLite)
============================================================
กฎการบันทึก:
  - เห็นหน้าพนักงานครั้งแรกของวัน  → บันทึกเป็น "เวลาเข้า"
  - เห็นหน้าครั้งต่อ ๆ มาในวันเดียวกัน → อัปเดต "เวลาออก" (ล่าสุด)
  → กล้องตัวเดียวพอ, พนักงานเดินผ่านเฉย ๆ ไม่ต้องกดปุ่ม
  - มี cooldown กันบันทึกรัว ๆ ตอนยืนหน้ากล้องนาน

คนแปลกหน้า (UNKNOWN) เก็บแยกในตาราง stranger_log ไม่ปนกับพนักงาน

ใช้ SQLite (ไฟล์เดียว, ไม่ต้องลง server, มากับ Python)
"""
from __future__ import annotations

import datetime
import sqlite3
import threading
from pathlib import Path


class AttendanceDB:
    def __init__(self, db_path: str, cooldown_sec: float = 60.0):
        self.path = Path(db_path)
        self.cooldown = cooldown_sec         # กันอัปเดตถี่เกินไปต่อคน (วินาที)
        self.lock = threading.Lock()
        self._last_mark: dict[str, float] = {}   # name -> last update ts (in-memory)
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(str(self.path))
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self.lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    name      TEXT NOT NULL,
                    date      TEXT NOT NULL,           -- YYYY-MM-DD
                    check_in  TEXT NOT NULL,           -- HH:MM:SS
                    check_out TEXT,                    -- HH:MM:SS (ล่าสุด)
                    in_snapshot  TEXT,
                    out_snapshot TEXT,
                    UNIQUE(name, date)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS stranger_log (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    time     TEXT NOT NULL,            -- ISO
                    snapshot TEXT
                )
            """)

    # ------------------------------------------------------------------
    def mark_seen(self, name: str, snapshot: str = "") -> dict | None:
        """
        เรียกเมื่อ detect เจอพนักงาน (name != UNKNOWN)
        คืน event dict ถ้ามีการบันทึกจริง (เข้า/ออก) หรือ None ถ้าอยู่ใน cooldown
        """
        import time as _t
        now_ts = _t.time()
        last = self._last_mark.get(name, 0)
        if now_ts - last < self.cooldown:
            return None                      # เพิ่งบันทึกไป ยังไม่ถึงเวลาอัปเดต
        self._last_mark[name] = now_ts

        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        clock = now.strftime("%H:%M:%S")

        with self.lock, self._conn() as c:
            row = c.execute(
                "SELECT * FROM attendance WHERE name=? AND date=?", (name, date)
            ).fetchone()

            if row is None:
                # ครั้งแรกของวัน → เวลาเข้า
                c.execute(
                    "INSERT INTO attendance (name,date,check_in,in_snapshot) VALUES (?,?,?,?)",
                    (name, date, clock, snapshot),
                )
                return {"type": "check_in", "name": name, "date": date, "time": clock}
            else:
                # เห็นอีกครั้ง → อัปเดตเวลาออก (ล่าสุด)
                c.execute(
                    "UPDATE attendance SET check_out=?, out_snapshot=? WHERE name=? AND date=?",
                    (clock, snapshot, name, date),
                )
                return {"type": "check_out", "name": name, "date": date, "time": clock}

    def log_stranger(self, snapshot: str = "") -> dict:
        """บันทึกคนแปลกหน้า (แยกจากพนักงาน)"""
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        with self.lock, self._conn() as c:
            c.execute("INSERT INTO stranger_log (time,snapshot) VALUES (?,?)",
                      (now_iso, snapshot))
        return {"type": "stranger", "time": now_iso, "snapshot": snapshot}

    # ------------------------------------------------------------------
    def get_attendance(self, date: str) -> list[dict]:
        """ดึงตารางเข้างานของวันนั้น พร้อมคำนวณชั่วโมงทำงาน"""
        with self.lock, self._conn() as c:
            rows = c.execute(
                "SELECT * FROM attendance WHERE date=? ORDER BY check_in", (date,)
            ).fetchall()

        out = []
        for r in rows:
            hours = ""
            if r["check_out"]:
                try:
                    fmt = "%H:%M:%S"
                    t_in = datetime.datetime.strptime(r["check_in"], fmt)
                    t_out = datetime.datetime.strptime(r["check_out"], fmt)
                    delta = t_out - t_in
                    secs = max(delta.total_seconds(), 0)
                    hours = f"{int(secs // 3600):02d}:{int((secs % 3600) // 60):02d}"
                except Exception:
                    hours = ""
            out.append({
                "name": r["name"],
                "date": r["date"],
                "check_in": r["check_in"],
                "check_out": r["check_out"] or "",
                "work_hours": hours,
                "in_snapshot": r["in_snapshot"] or "",
                "out_snapshot": r["out_snapshot"] or "",
            })
        return out

    def get_strangers(self, date: str) -> list[dict]:
        """ดึง log คนแปลกหน้าของวันนั้น"""
        with self.lock, self._conn() as c:
            rows = c.execute(
                "SELECT * FROM stranger_log WHERE time LIKE ? ORDER BY time DESC",
                (f"{date}%",),
            ).fetchall()
        return [{"time": r["time"], "snapshot": r["snapshot"] or ""} for r in rows]

    def summary(self, date: str) -> dict:
        """สรุปของวัน: จำนวนพนักงานที่มา / ยังไม่ออก / คนแปลกหน้า"""
        with self.lock, self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) FROM attendance WHERE date=?", (date,)
            ).fetchone()[0]
            not_out = c.execute(
                "SELECT COUNT(*) FROM attendance WHERE date=? AND (check_out IS NULL OR check_out='')",
                (date,),
            ).fetchone()[0]
            strangers = c.execute(
                "SELECT COUNT(*) FROM stranger_log WHERE time LIKE ?", (f"{date}%",)
            ).fetchone()[0]
        return {
            "present": total,
            "still_in": not_out,
            "strangers": strangers,
        }

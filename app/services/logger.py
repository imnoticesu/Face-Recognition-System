"""
Event logger — บันทึกการตรวจจับลงไฟล์ json แบบ thread-safe
"""
from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path


class EventLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.lock = threading.Lock()
        try:
            self.entries = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        except Exception:
            self.entries = []

    def log(self, name: str, snap: str, status: str) -> dict:
        entry = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "name": name,
            "snapshot": snap,
            "status": status,  # "known" | "unknown"
        }
        with self.lock:
            self.entries.append(entry)
            try:
                self.path.write_text(
                    json.dumps(self.entries, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[Logger] เขียนไฟล์ไม่ได้: {e}")
        print(f"[LOG] {entry['time']} | {name} → {snap or '(ซ้ำ, ไม่ snap)'}")
        return entry

    def between(self, start_iso: str, end_iso: str) -> list[dict]:
        with self.lock:
            data = list(self.entries)
        return [e for e in data if start_iso <= e["time"] <= end_iso]

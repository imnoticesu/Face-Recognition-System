"""
Shared state ระหว่าง thread ต่างๆ (capture / worker / API)
รวมไว้ที่เดียว พร้อม lock — แทนการกระจาย global ทั่วไฟล์
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class SharedState:
    def __init__(self):
        # เฟรม
        self._latest_frame: Optional[np.ndarray] = None      # เฟรมดิบสำหรับ worker
        self._frame_seq: int = 0                             # นับเฟรมใหม่ ให้ worker เช็คว่าซ้ำไหม
        self._latest_annotated: Optional[np.ndarray] = None  # เฟรมวาดกรอบสำหรับสตรีม
        self._frame_lock = threading.Lock()
        self._annotated_lock = threading.Lock()

        # ผลตรวจจับล่าสุด (กรอบ+ชื่อ)
        self._results: list[dict] = []
        self._result_lock = threading.Lock()

        # คนที่อยู่หน้ากล้องตอนนี้ {name: last_seen_ts}
        self._in_view: dict[str, float] = {}
        self._in_view_lock = threading.Lock()

        # สถานะระบบ
        self.running = True
        self.camera_connected = False
        self.current_fps = 0.0

    # ---- เฟรมดิบ ----
    def set_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self._latest_frame = frame
            self._frame_seq += 1

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_frame_with_seq(self):
        """คืน (frame, seq) เพื่อให้ worker เช็คว่าเป็นเฟรมใหม่ไหม"""
        with self._frame_lock:
            if self._latest_frame is None:
                return None, self._frame_seq
            return self._latest_frame.copy(), self._frame_seq

    # ---- เฟรมวาดกรอบ ----
    def set_annotated(self, frame: np.ndarray):
        with self._annotated_lock:
            self._latest_annotated = frame

    def get_annotated(self) -> Optional[np.ndarray]:
        with self._annotated_lock:
            return self._latest_annotated.copy() if self._latest_annotated is not None else None

    # ---- ผลตรวจจับ ----
    def set_results(self, results: list[dict]):
        with self._result_lock:
            self._results = results

    def get_results(self) -> list[dict]:
        with self._result_lock:
            return list(self._results)

    # ---- คนหน้ากล้อง ----
    def mark_in_view(self, name: str):
        with self._in_view_lock:
            self._in_view[name] = time.time()

    def expire_in_view(self, timeout: float):
        now = time.time()
        with self._in_view_lock:
            for n in [n for n, t in self._in_view.items() if now - t > timeout]:
                del self._in_view[n]

    def in_view_count(self) -> int:
        with self._in_view_lock:
            return len(self._in_view)


state = SharedState()

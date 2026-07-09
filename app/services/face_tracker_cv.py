"""
CV Face Tracker — ตามใบหน้าทุกเฟรมด้วย OpenCV tracker (KCF)
============================================================
แนวคิด: detector หนัก (yunet+DeepFace) รันเป็นครั้งคราวใน worker
        ส่วน tracker เบา ๆ ตัวนี้ตามหน้าให้ "ทุกเฟรมของกล้อง"
        → กรอบลื่น + ตามทันแม้ขยับเร็ว โดยไม่ต้อง detect ถี่

ทำงาน 2 ขา:
  - worker เรียก reset_tracks() เมื่อ detect รอบใหม่เสร็จ
    (ให้ตำแหน่ง+ชื่อที่ยืนยันแล้ว เป็นจุดตั้งต้นของ tracker)
  - capture loop เรียก update() ทุกเฟรม
    (tracker ประเมินตำแหน่งใหม่จากภาพจริง แล้วคืนกรอบที่จะวาด)

ถ้า opencv ไม่มี tracker (รุ่นเก่า) จะคืน None ให้ pipeline
fallback ไปใช้ RenderSmoother เดิมแทน
"""
from __future__ import annotations

import threading
import cv2


def _make_tracker():
    """สร้าง tracker รองรับทั้ง opencv ใหม่/เก่า คืน None ถ้าไม่มี
    เรียงตามความเร็ว: MOSSE (เร็วสุด ~10 เท่าของ CSRT) → KCF → CSRT
    งาน realtime บน CPU ควรใช้ MOSSE เพราะ detect รอบใหม่จะแก้ตำแหน่งให้เองอยู่แล้ว
    """
    factories = [
        lambda: cv2.legacy.TrackerMOSSE_create() if hasattr(cv2, "legacy") else None,
        lambda: cv2.TrackerMOSSE_create() if hasattr(cv2, "TrackerMOSSE_create") else None,
        lambda: cv2.TrackerKCF_create(),
        lambda: cv2.legacy.TrackerKCF_create() if hasattr(cv2, "legacy") else None,
        lambda: cv2.TrackerCSRT_create(),
    ]
    for factory in factories:
        try:
            t = factory()
            if t is not None:
                return t
        except Exception:
            continue
    return None


def trackers_available() -> bool:
    return _make_tracker() is not None


class CVFaceTracker:
    TRACK_SCALE = 0.5   # ตามหน้าบนภาพย่อครึ่ง → เร็วขึ้น ~4 เท่า แล้วขยายกรอบกลับ

    def __init__(self):
        self._lock = threading.Lock()
        # แต่ละ track: {"tracker": obj, "name": str, "box": (x,y,w,h)}
        self._tracks: list[dict] = []

    def _small(self, frame):
        s = self.TRACK_SCALE
        return cv2.resize(frame, (0, 0), fx=s, fy=s)

    def reset_tracks(self, frame, detections: list[dict]):
        """
        worker เรียกเมื่อ detect รอบใหม่เสร็จ
        frame: เฟรมเต็มความละเอียด (BGR) ที่ detect มาจากมัน
        detections: [{"box": (x,y,w,h), "name": str}, ...]
        สร้าง tracker ใหม่จากตำแหน่งที่ detect ได้ (แม่นกว่าเดา)
        """
        s = self.TRACK_SCALE
        small = self._small(frame)
        new_tracks = []
        for det in detections:
            x, y, w, h = det["box"]
            if w <= 0 or h <= 0:
                continue
            t = _make_tracker()
            if t is None:
                continue
            try:
                t.init(small, (int(x * s), int(y * s), int(w * s), int(h * s)))
                new_tracks.append({"tracker": t, "name": det["name"],
                                   "box": (int(x), int(y), int(w), int(h))})
            except Exception:
                continue
        with self._lock:
            self._tracks = new_tracks

    def update(self, frame) -> list[dict]:
        """
        capture loop เรียกทุกเฟรม
        อัปเดตตำแหน่งทุก track จากภาพจริง คืนกรอบที่จะวาด
        track ที่หลุดจะคงกรอบเดิมไว้ไม่กี่เฟรม (กันกระพริบ) ก่อนตัดทิ้ง
        """
        MAX_MISS = 8  # จำนวนเฟรมที่ยอมให้ tracker หลุดก่อนตัดทิ้ง
        s = self.TRACK_SCALE
        small = self._small(frame)
        out = []
        with self._lock:
            tracks = list(self._tracks)

        alive = []
        for tr in tracks:
            try:
                ok, box = tr["tracker"].update(small)
            except Exception:
                ok = False
                box = None
            if ok and box is not None:
                x, y, w, h = box
                tr["box"] = (int(x / s), int(y / s), int(w / s), int(h / s))
                tr["miss"] = 0
                out.append({"box": tr["box"], "name": tr["name"]})
                alive.append(tr)
            else:
                # tracker หลุด — คงกรอบเดิมไว้ชั่วคราว กันกรอบกระพริบหาย
                tr["miss"] = tr.get("miss", 0) + 1
                if tr["miss"] <= MAX_MISS:
                    out.append({"box": tr["box"], "name": tr["name"]})
                    alive.append(tr)

        with self._lock:
            if len(self._tracks) == len(tracks):
                self._tracks = alive
        return out

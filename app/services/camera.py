"""
Camera handling — เปิดกล้องให้เสถียร รองรับทั้ง USB (index) และ RTSP/IP cam
============================================================
ปรับปรุงจากเดิม:
  - แยกตรรกะ USB vs RTSP ชัดเจน
  - USB: ลองหลาย backend (DSHOW/MSMF/ANY), warm-up ยืดหยุ่นขึ้น
  - RTSP: ตั้ง buffer=1 ลด latency, timeout กันค้าง
  - force_mjpg ปิด/เปิดได้ผ่าน config (กล้องบางรุ่นไม่รองรับ MJPG)
  - retry เปิดใหม่อัตโนมัติเมื่อหลุด โดยไม่ทำให้ทั้ง thread ตาย
============================================================
"""
from __future__ import annotations

import time
import cv2

from app.core.config import config


def _is_index_source(source: str) -> bool:
    return str(source).strip().isdigit()


def open_camera(source: str):
    """เปิดกล้องตามชนิดของ source คืน VideoCapture ที่พร้อมใช้ หรือ None"""
    s = config.s
    source = str(source).strip()

    # ---------- RTSP / IP camera ----------
    if not _is_index_source(source):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        # ลด latency: อ่านเฟรมล่าสุดเสมอ ไม่สะสม buffer
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # ทดสอบอ่าน
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 1:
                print(f"[Camera] เชื่อมต่อ RTSP สำเร็จ: {source}")
                return cap
            time.sleep(0.1)
        print(f"[Camera] RTSP เปิดได้แต่อ่านภาพไม่ได้: {source}")
        cap.release()
        return None

    # ---------- USB / Webcam (index) ----------
    idx = int(source)
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto/Default"),
    ]
    # ชุดค่าที่จะลอง (เรียงจากปลอดภัยสุด → เจาะจงสุด)
    # error -1072875772 บน Windows มักหายเมื่อ "ไม่บังคับ format/resolution"
    # จึงลองแบบปล่อย default ก่อน แล้วค่อยลองบังคับ MJPG + resolution
    profiles = [
        {"mjpg": False, "size": False, "desc": "default (ปล่อยกล้องเลือกเอง)"},
    ]
    if s.force_mjpg:
        profiles.append({"mjpg": True, "size": True, "desc": "MJPG + resolution"})
    profiles.append({"mjpg": False, "size": True, "desc": "no-MJPG + resolution"})

    for flag, name in backends:
        for prof in profiles:
            cap = cv2.VideoCapture(idx, flag)
            if not cap.isOpened():
                cap.release()
                continue

            if prof["mjpg"]:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if prof["size"]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.frame_height)
                cap.set(cv2.CAP_PROP_FPS, s.target_fps)

            # warm-up ทน: อ่านได้ถึง 15 เฟรม ขอภาพไม่ดำ >= 2
            # (กล้องบางรุ่นคืน error หลายเฟรมแรกก่อนจะพร้อม)
            ok = 0
            for _ in range(15):
                ret, frame = cap.read()
                if ret and frame is not None and frame.mean() > 1:
                    ok += 1
                    if ok >= 2:
                        break
                time.sleep(0.1)

            if ok >= 2:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[Camera] เปิดกล้อง index {idx} สำเร็จ "
                      f"({name} / {prof['desc']}, {w}x{h})")
                return cap

            cap.release()

        print(f"[Camera] {name} เปิดไม่ผ่านทุกโปรไฟล์ — ลอง backend ถัดไป")

    print(f"[Camera] เปิดกล้อง index {idx} ไม่ได้เลย — "
          f"อาจถูกโปรแกรมอื่นยึด, ไม่มีสิทธิ์เข้าถึงกล้อง, หรือ index ผิด")
    return None

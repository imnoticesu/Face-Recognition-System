"""
วาดกรอบใบหน้า + HUD ลงบนเฟรม
"""
from __future__ import annotations

import datetime
import cv2

COLOR_KNOWN = (0, 220, 100)
COLOR_UNKNOWN = (0, 60, 220)
COLOR_HUD = (200, 200, 200)


def draw_box(frame, x, y, w, h, name, color):
    r = 14
    for px, py in [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]:
        cv2.line(frame, (px, py), (px + (r if px == x else -r), py), color, 2)
        cv2.line(frame, (px, py), (px, py + (r if py == y else -r)), color, 2)
    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), (20, 20, 20), -1)
    cv2.putText(frame, name, (x + 5, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)


def draw_hud(frame, fps, face_n):
    now = datetime.datetime.now()
    year_be = now.year + 543          # ค.ศ. → พ.ศ.
    ts = now.strftime(f"%d/%m/{year_be}  %H:%M:%S")

    h, w = frame.shape[:2]
    # ปรับขนาดตัวอักษรตามความกว้างภาพ (ฐาน 640px) กันตัวใหญ่/ซ้อนบนภาพเล็ก
    scale = max(0.4, min(0.6, w / 1280.0))
    thick_bg = 3
    line_gap = int(26 * (w / 640.0))          # ระยะห่างบรรทัดขยับตามขนาดภาพ
    x = 12
    y0 = int(22 * (w / 640.0))

    lines = [f"AI CCTV  |  {ts}", f"FPS: {fps:.1f}   Faces: {face_n}"]
    for i, line in enumerate(lines):
        y = y0 + i * line_gap
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thick_bg, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_HUD,
                    1, cv2.LINE_AA)

    cv2.putText(frame, "LIVE", (x, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, scale - 0.04), (0, 200, 80), 1, cv2.LINE_AA)


def color_for(name: str):
    return COLOR_KNOWN if name != "UNKNOWN" else COLOR_UNKNOWN

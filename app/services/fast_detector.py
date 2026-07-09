"""
Fast Detector — หาตำแหน่งใบหน้าด้วย YuNet ผ่าน OpenCV โดยตรง (ไม่ผ่าน DeepFace)
============================================================
ทำไมเร็วกว่าเดิม:
  - เรียก cv2.FaceDetectorYN ตรง ๆ ไม่มี overhead ของ DeepFace
    (DeepFace ห่อหลายชั้น + แปลงภาพซ้ำซ้อน ทำให้ detect ช้ากว่าที่ควรมาก)
  - โมเดล YuNet เล็กมาก (~230KB) เร็วระดับ ~5-15ms บน CPU
  - จับหน้าเอียง/มุมข้างได้ดี

โมเดล: ดาวน์โหลดอัตโนมัติครั้งแรก (ต่อเน็ตครั้งเดียว) เก็บไว้ข้างโปรเจกต์
ถ้าโหลดไม่ได้/ไม่มีเน็ต → available=False ให้ pipeline fallback ไป DeepFace detector
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_MODEL_URLS = [
    # ลิงก์ LFS media (ไฟล์จริง)
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    # สำรอง
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
_MODEL_NAME = "face_detection_yunet_2023mar.onnx"


def _model_path() -> Path:
    # เก็บไว้ที่โฟลเดอร์โปรเจกต์ (ข้าง backend.py)
    base = Path(__file__).resolve().parent.parent.parent
    return base / "models" / _MODEL_NAME


def _ensure_model() -> Path | None:
    p = _model_path()
    if p.exists() and p.stat().st_size > 100_000:   # ไฟล์จริง ~230KB (กัน pointer LFS)
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    for url in _MODEL_URLS:
        try:
            print(f"[FastDetector] กำลังโหลดโมเดล YuNet... ({url.split('/')[2]})")
            urllib.request.urlretrieve(url, str(p))
            if p.stat().st_size > 100_000:
                print(f"[FastDetector] โหลดโมเดลสำเร็จ ({p.stat().st_size//1024} KB)")
                return p
        except Exception as e:
            print(f"[FastDetector] โหลดไม่สำเร็จ: {e}")
    if p.exists():
        p.unlink(missing_ok=True)
    return None


class FastDetector:
    def __init__(self, score_thresh: float = 0.7):
        self.available = False
        self._det = None
        self._size = (0, 0)
        if not hasattr(cv2, "FaceDetectorYN_create"):
            print("[FastDetector] opencv รุ่นนี้ไม่มี FaceDetectorYN — fallback DeepFace")
            return
        model = _ensure_model()
        if model is None:
            print("[FastDetector] ไม่มีโมเดล YuNet — fallback DeepFace detector")
            return
        try:
            self._det = cv2.FaceDetectorYN_create(
                str(model), "", (320, 240),
                score_threshold=score_thresh,
                nms_threshold=0.3,
                top_k=50,
            )
            self.available = True
            print("[FastDetector] ใช้ YuNet ผ่าน OpenCV โดยตรง (เร็ว, จับหน้าเอียงได้)")
        except Exception as e:
            print(f"[FastDetector] เปิด YuNet ไม่ได้: {e}")

    def detect(self, frame_bgr) -> list[dict]:
        """คืน [{"box": (x,y,w,h), "confidence": float}] พิกัดบนภาพที่ส่งเข้า"""
        if not self.available or self._det is None:
            return []
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._size:
            self._det.setInputSize((w, h))
            self._size = (w, h)
        try:
            _, faces = self._det.detect(frame_bgr)
        except Exception:
            return []
        out = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                if bw <= 0 or bh <= 0:
                    continue
                out.append({
                    "box": (max(x, 0), max(y, 0), bw, bh),
                    "confidence": float(f[14]) if len(f) > 14 else 1.0,
                })
        return out

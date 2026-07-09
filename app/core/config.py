"""
Config รวมศูนย์ — โหลด/เซฟจาก config.json แบบ thread-safe
ใช้ dataclass เพื่อให้ type ชัดเจนและแก้ค่าได้ปลอดภัย
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.json"


@dataclass
class Settings:
    # ---- กล้อง ----
    camera_source: str = "0"          # "0"/"1"/... สำหรับ USB หรือ "rtsp://..." สำหรับ IP cam
    frame_width: int = 640            # ลดจาก 1280 → detect เร็วขึ้นมากบน CPU
    frame_height: int = 480           # ลดจาก 720
    target_fps: int = 30
    force_mjpg: bool = True           # บังคับ MJPG (ช่วยกล้อง USB บางรุ่น) — ปิดได้ถ้ากล้องไม่รองรับ

    # ---- การประมวลผล ----
    model: str = "Facenet"
    detector: str = "yunet"           # เร็วกว่า opencv + จับหน้ามุมข้าง/เอียงได้ (โหลด model ครั้งแรกต้องต่อเน็ต)
    process_every: int = 2            # mediapipe เบามาก detect เกือบทุกเฟรมได้ = กรอบลื่น
    scale: float = 0.5                # ย่อภาพก่อน detect (ตอนนี้ 640x480 → detect ที่ 320x240)
    distance_thresh: float = 0.55     # cosine distance ต่ำกว่านี้ = ตรงกัน
    min_confidence: float = 0.90
    min_face_px: int = 30

    # ---- ตรรกะเหตุการณ์ ----
    leave_timeout: float = 5.0
    dedup_thresh: float = 0.45
    snap_unknown: bool = True
    smooth_alpha: float = 0.5         # กรอบตามหน้าไวขึ้น (ต่ำ=หนืด สูง=ไวแต่กระตุก)

    # ---- สตรีม ----
    jpeg_quality: int = 70            # เบาลงนิด ส่งภาพไวขึ้น
    stream_fps: int = 30              # สตรีมลื่นขึ้น (เดิม 20)

    # ---- พาธ (ไม่เซฟลง config.json) ----
    known_faces_dir: str = field(default_factory=lambda: str(BASE_DIR / "known_faces"))
    snapshots_dir: str = field(default_factory=lambda: str(BASE_DIR / "snapshots"))
    log_file: str = field(default_factory=lambda: str(BASE_DIR / "cctv_log.json"))

    # ฟิลด์ที่อนุญาตให้แก้ผ่าน API/บันทึกลงไฟล์
    _PERSIST = (
        "camera_source", "frame_width", "frame_height", "target_fps", "force_mjpg",
        "model", "detector", "process_every", "scale", "distance_thresh",
        "min_confidence", "min_face_px", "leave_timeout", "dedup_thresh",
        "snap_unknown", "smooth_alpha", "jpeg_quality", "stream_fps",
    )

    def public_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._PERSIST}


class ConfigManager:
    """ห่อ Settings ด้วย lock ให้แก้ได้จากหลาย thread อย่างปลอดภัย"""

    def __init__(self):
        self._lock = threading.RLock()
        self._settings = Settings()
        self._load()

    @property
    def s(self) -> Settings:
        # อ่านค่าตรงๆ ได้ (การอ่าน field เดี่ยวเป็น atomic พอสำหรับ use case นี้)
        return self._settings

    def _load(self):
        if not CONFIG_PATH.exists():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            with self._lock:
                for k, v in data.items():
                    if k in Settings._PERSIST:
                        setattr(self._settings, k, v)
            print("[Config] โหลด config.json สำเร็จ")
        except Exception as e:
            print(f"[Config] โหลด config.json ไม่ได้: {e}")

    def save(self):
        try:
            CONFIG_PATH.write_text(
                json.dumps(self._settings.public_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("[Config] บันทึก config.json สำเร็จ")
        except Exception as e:
            print(f"[Config] บันทึกไม่ได้: {e}")

    def update(self, changes: dict) -> dict:
        """อัปเดตเฉพาะฟิลด์ที่อนุญาต แล้วเซฟ คืนค่า config ปัจจุบัน"""
        with self._lock:
            for k, v in changes.items():
                if k in Settings._PERSIST and v is not None:
                    setattr(self._settings, k, v)
        self.save()
        return self._settings.public_dict()


config = ConfigManager()

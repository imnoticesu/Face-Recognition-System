"""
Face Engine — แก้คอขวดหลักของระบบเดิม
============================================================
ปัญหาเดิม: ใช้ DeepFace.verify() วนเทียบทีละไฟล์ในลูป detection
           → 100+ คน = ช้ามาก, โหลดโมเดลซ้ำทุกครั้ง, ไม่ realtime

วิธีใหม่:
  1. โหลดโมเดลครั้งเดียว (represent ใช้โมเดลที่ warm ไว้)
  2. คำนวณ embedding ของ known faces ทั้งหมด "ครั้งเดียว" ตอนเริ่ม
     แล้ว cache ไว้เป็น matrix (N_known x D)
  3. ตอน detect: หา embedding ของหน้าที่เจอ 1 ครั้ง แล้ว
     เทียบ cosine กับ matrix ทั้งก้อนด้วย numpy (เร็วมาก แม้ 100+ คน)
  4. เฝ้าดูโฟลเดอร์ known_faces — ถ้ามีไฟล์เพิ่ม/ลบ ค่อย rebuild cache
============================================================
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARN] ไม่พบ deepface — โหมดไม่ตรวจจับ (pip install deepface tf-keras)")


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-10)


class FaceEngine:
    def __init__(self, model_name: str, detector: str, known_dir: str):
        self.model_name = model_name
        self.detector = detector
        self.known_dir = Path(known_dir)
        self.known_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._names: list[str] = []          # ชื่อคนของแต่ละ embedding
        self._matrix: np.ndarray | None = None  # (N x D) embeddings ที่ normalize แล้ว
        self._signature: str = ""            # ลายเซ็นสถานะโฟลเดอร์ ไว้เช็คว่าต้อง rebuild ไหม

        if DEEPFACE_AVAILABLE:
            self._warmup()
            self.rebuild_cache()

    # ------------------------------------------------------------------
    def _warmup(self):
        """โหลดโมเดลเข้าเมมครั้งเดียว เพื่อไม่ให้ครั้งแรกที่ detect ช้า"""
        try:
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(
                img_path=dummy, model_name=self.model_name,
                detector_backend="skip", enforce_detection=False,
            )
            print(f"[Engine] warmup โมเดล {self.model_name} เรียบร้อย")
        except Exception as e:
            print(f"[Engine] warmup ล้มเหลว: {e}")

    # ------------------------------------------------------------------
    def _folder_signature(self) -> str:
        """สร้างลายเซ็นจากชื่อไฟล์+เวลาแก้ไข เพื่อรู้ว่าโฟลเดอร์เปลี่ยนไหม"""
        parts = []
        for p in sorted(self.known_dir.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    parts.append(f"{p}:{p.stat().st_mtime_ns}")
                except OSError:
                    continue
        return "|".join(parts)

    def maybe_rebuild(self):
        """เรียกเป็นระยะ — rebuild เฉพาะเมื่อโฟลเดอร์เปลี่ยนจริง"""
        if not DEEPFACE_AVAILABLE:
            return
        sig = self._folder_signature()
        if sig != self._signature:
            print("[Engine] ตรวจพบการเปลี่ยนแปลงใน known_faces → rebuild cache")
            self.rebuild_cache(signature=sig)

    def rebuild_cache(self, signature: str | None = None):
        """คำนวณ embedding ของ known faces ทั้งหมดครั้งเดียว แล้วเก็บเป็น matrix"""
        if not DEEPFACE_AVAILABLE:
            return
        names: list[str] = []
        vecs: list[np.ndarray] = []

        imgs = sorted(
            list(self.known_dir.rglob("*.jpg"))
            + list(self.known_dir.rglob("*.jpeg"))
            + list(self.known_dir.rglob("*.png"))
        )
        for img_path in imgs:
            person = img_path.parent.name if img_path.parent != self.known_dir else img_path.stem
            try:
                reps = DeepFace.represent(
                    img_path=str(img_path), model_name=self.model_name,
                    detector_backend=self.detector, enforce_detection=False, align=True,
                )
                if not reps:
                    continue
                emb = np.asarray(reps[0]["embedding"], dtype=np.float32)
                vecs.append(emb)
                names.append(person)
            except Exception as e:
                print(f"[Engine] embed ไม่ได้ {img_path.name}: {e}")

        with self._lock:
            if vecs:
                self._matrix = _l2_normalize(np.vstack(vecs))
                self._names = names
            else:
                self._matrix = None
                self._names = []
            self._signature = signature if signature is not None else self._folder_signature()

        print(f"[Engine] cache พร้อม: {len(names)} รูป จาก known_faces")

    # ------------------------------------------------------------------
    def embed_face(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """หา embedding ของภาพใบหน้าที่ crop มาแล้ว (ข้าม detector ในตัว)"""
        if not DEEPFACE_AVAILABLE:
            return None
        try:
            reps = DeepFace.represent(
                img_path=face_bgr, model_name=self.model_name,
                detector_backend="skip", enforce_detection=False, align=False,
            )
            if not reps:
                return None
            return np.asarray(reps[0]["embedding"], dtype=np.float32)
        except Exception:
            return None

    def identify(self, face_bgr: np.ndarray, distance_thresh: float) -> tuple[str, float]:
        """
        เทียบใบหน้ากับ known ทั้งหมดในครั้งเดียวด้วย numpy
        คืน (ชื่อ, cosine_distance). ถ้าไม่เจอคืน ("UNKNOWN", 1.0)
        """
        emb = self.embed_face(face_bgr)
        if emb is None:
            return "UNKNOWN", 1.0

        with self._lock:
            matrix = self._matrix
            names = self._names

        if matrix is None or len(names) == 0:
            return "UNKNOWN", 1.0

        q = _l2_normalize(emb.reshape(1, -1))          # (1 x D)
        sims = (matrix @ q.T).ravel()                  # cosine similarity กับทุก known
        best_idx = int(np.argmax(sims))
        best_dist = 1.0 - float(sims[best_idx])        # cosine distance

        if best_dist < distance_thresh:
            return names[best_idx], best_dist
        return "UNKNOWN", best_dist

    def is_duplicate(self, face_bgr: np.ndarray, folder: Path, dedup_thresh: float) -> bool:
        """เช็คว่าหน้านี้ซ้ำกับสแนปช็อตในโฟลเดอร์ไหม (ใช้ embedding เทียบ ไม่วน verify)"""
        if not DEEPFACE_AVAILABLE:
            return False
        saved = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if not saved:
            return False
        emb = self.embed_face(face_bgr)
        if emb is None:
            return False
        q = _l2_normalize(emb.reshape(1, -1))
        for sp in saved:
            try:
                reps = DeepFace.represent(
                    img_path=str(sp), model_name=self.model_name,
                    detector_backend="skip", enforce_detection=False, align=False,
                )
                if not reps:
                    continue
                other = _l2_normalize(np.asarray(reps[0]["embedding"], dtype=np.float32).reshape(1, -1))
                dist = 1.0 - float((q @ other.T).ravel()[0])
                if dist < dedup_thresh:
                    return True
            except Exception:
                continue
        return False

    def detect_faces(self, img_bgr: np.ndarray) -> list[dict]:
        """หาใบหน้าในภาพ คืน list ของ facial_area + confidence"""
        if not DEEPFACE_AVAILABLE:
            return []
        try:
            return DeepFace.extract_faces(
                img_path=img_bgr, detector_backend=self.detector,
                enforce_detection=False, align=True,
            )
        except Exception as e:
            print(f"[Engine] detect error: {e}")
            return []

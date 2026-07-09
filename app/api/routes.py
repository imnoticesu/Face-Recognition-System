"""
API routes — endpoint ทั้งหมด แยกจากตรรกะประมวลผล
"""
from __future__ import annotations

import base64
import datetime
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from app.core.config import config
from app.core.state import state
from app.services import pipeline
from app.services.face_engine import DEEPFACE_AVAILABLE
from app.services.websocket import manager

router = APIRouter()


# ---------- MJPEG STREAM ----------
def mjpeg_generator():
    boundary = b"--frame"
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Camera not available", (140, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    while True:
        frame = state.get_annotated()
        if frame is None:
            frame = placeholder.copy()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.s.jpeg_quality])
        if not ok:
            continue
        yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1 / max(config.s.stream_fps, 1))


@router.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ---------- STATUS ----------
@router.get("/api/status")
def get_status():
    return {
        "camera_connected": state.camera_connected,
        "fps": round(state.current_fps, 1),
        "in_camera_view": state.in_view_count(),
        "deepface_available": DEEPFACE_AVAILABLE,
    }


# ---------- LOGS ----------
@router.get("/api/logs")
def get_logs(start_date: str, end_date: str,
             start_time: str = "00:00:00", end_time: str = "23:59:59"):
    start_iso = f"{start_date}T{start_time}"
    end_iso = f"{end_date}T{end_time}"
    entries = pipeline.event_logger.between(start_iso, end_iso)
    total = len(entries)
    unknown = sum(1 for e in entries if e["status"] == "unknown")
    known = sum(1 for e in entries if e["status"] == "known")
    return {
        "summary": {
            "total_captured": total,
            "visitors_unknown": unknown,
            "in_camera_view": state.in_view_count(),
            "staff_identified": known,
        },
        "entries": list(reversed(entries)),
    }


@router.get("/api/snapshot/{name}/{filename}")
def get_snapshot(name: str, filename: str):
    path = Path(config.s.snapshots_dir) / name / filename
    if not path.exists():
        return {"error": "not found"}
    return FileResponse(path)


# ---------- REGISTER ----------
class RegisterRequest(BaseModel):
    name: str
    images_base64: list[str]


@router.post("/api/register")
def register_face(req: RegisterRequest):
    if not req.name.strip():
        return {"success": False, "error": "กรุณาระบุชื่อ"}
    person_dir = Path(config.s.known_faces_dir) / req.name.strip()
    person_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, b64 in enumerate(req.images_base64):
        try:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(person_dir / f"{req.name}_{ts}_{i}.jpg"), img)
            saved += 1
        except Exception as e:
            print(f"[Register] error saving image {i}: {e}")

    # rebuild cache ทันทีหลังลงทะเบียนเสร็จ เพื่อให้ detect คนใหม่ได้เลย
    if saved > 0 and pipeline.engine is not None:
        pipeline.engine.rebuild_cache()

    return {"success": saved > 0, "saved_count": saved, "folder": str(person_dir)}


@router.get("/api/known_faces")
def list_known_faces():
    known_dir = Path(config.s.known_faces_dir)
    people = []
    if known_dir.exists():
        for d in sorted(known_dir.iterdir()):
            if d.is_dir():
                imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
                people.append({
                    "name": d.name,
                    "photo_count": len(imgs),
                    "first_image": imgs[0].name if imgs else None,
                })
    return {"people": people}


@router.get("/api/known_face_image/{name}/{filename}")
def get_known_face_image(name: str, filename: str):
    path = Path(config.s.known_faces_dir) / name / filename
    if not path.exists():
        return {"error": "not found"}
    return FileResponse(path)


@router.delete("/api/known_faces/{name}")
def delete_known_face(name: str):
    person_dir = Path(config.s.known_faces_dir) / name
    if person_dir.exists() and person_dir.is_dir():
        try:
            shutil.rmtree(person_dir)
            if pipeline.engine is not None:
                pipeline.engine.rebuild_cache()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "ไม่พบรายชื่อในระบบ"}


# ---------- CONFIG ----------
@router.get("/api/config")
def get_config():
    return config.s.public_dict()


class ConfigUpdate(BaseModel):
    camera_source: str | None = None
    distance_thresh: float | None = None
    min_confidence: float | None = None
    leave_timeout: float | None = None
    snap_unknown: bool | None = None
    smooth_alpha: float | None = None
    process_every: int | None = None
    scale: float | None = None
    jpeg_quality: int | None = None
    stream_fps: int | None = None
    force_mjpg: bool | None = None


@router.post("/api/config")
def update_config(update: ConfigUpdate):
    changes = {k: v for k, v in update.dict().items() if v is not None}
    updated = config.update(changes)
    return {"success": True, "config": updated}


# ---------- ATTENDANCE (บันทึกเวลาเข้า/ออกงาน) ----------
@router.get("/api/attendance")
def get_attendance(date: str | None = None):
    """ตารางเข้างานของวันที่ระบุ (default = วันนี้)"""
    if not date:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    records = pipeline.attendance.get_attendance(date)
    summary = pipeline.attendance.summary(date)
    return {"date": date, "summary": summary, "records": records}


@router.get("/api/attendance/strangers")
def get_strangers(date: str | None = None):
    """log คนแปลกหน้าของวันที่ระบุ"""
    if not date:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    return {"date": date, "strangers": pipeline.attendance.get_strangers(date)}


# ---------- WEBSOCKET ----------
@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

"""
Pipeline — 2 thread หลักของระบบ
============================================================
  capture_loop     : อ่านกล้องตลอดเวลา, วาดกรอบจากผลล่าสุด, ทำ annotated frame
  detection_worker : หยิบเฟรมดิบมา detect + identify (ใช้ embedding cache)
                     แล้ว push ผลกลับให้ capture_loop วาด

จุดที่เปลี่ยนจากเดิม:
  - identify ผ่าน FaceEngine (numpy cosine กับ cache) แทน verify วนไฟล์
  - dedup snapshot ใช้ embedding แทน verify
  - rebuild cache อัตโนมัติเมื่อ known_faces เปลี่ยน (ทุกๆ N รอบ)
============================================================
"""
from __future__ import annotations

import datetime
import time
from pathlib import Path

import cv2

from app.core.config import config
from app.core.state import state
from app.services import drawing
from app.services.camera import open_camera
from app.services.face_engine import FaceEngine, DEEPFACE_AVAILABLE
from app.services.logger import EventLogger
from app.services.tracking import FaceTracker, BoxSmoother, RenderSmoother
from app.services.face_tracker_cv import CVFaceTracker, trackers_available
from app.services.attendance import AttendanceDB
from app.services.fast_detector import FastDetector


# ---- อ็อบเจกต์ที่ pipeline ใช้ (ประกอบตอน start) ----
engine: FaceEngine | None = None
event_logger: EventLogger | None = None
tracker: FaceTracker | None = None
smoother: BoxSmoother | None = None
render_smoother: RenderSmoother | None = None
cv_tracker: CVFaceTracker | None = None
use_cv_tracker = False       # เปิดใช้ถ้า opencv มี tracker
attendance: AttendanceDB | None = None
fast_detector: FastDetector | None = None
IDENTIFY_EVERY = 10          # ระบุชื่อ (DeepFace) ทุกๆ N รอบ detect — ระหว่างนั้นใช้ชื่อเดิม
_identify_counter = [0]
_name_cache: list = []       # [(cx, cy, name)] ตำแหน่ง+ชื่อล่าสุดที่ระบุแล้ว
broadcaster = None  # ตั้งจากภายนอก (ConnectionManager) เพื่อ push websocket
_last_frame_seq = [-1]  # จำ seq เฟรมล่าสุดที่ worker detect ไปแล้ว


def init_pipeline(ws_broadcaster=None):
    global engine, event_logger, tracker, smoother, render_smoother, broadcaster
    global cv_tracker, use_cv_tracker, attendance, fast_detector
    s = config.s
    engine = FaceEngine(s.model, s.detector, s.known_faces_dir)
    event_logger = EventLogger(s.log_file)
    attendance = AttendanceDB(str(Path(s.log_file).parent / "attendance.db"))
    fast_detector = FastDetector()
    tracker = FaceTracker(leave_timeout=s.leave_timeout)
    smoother = BoxSmoother(alpha=s.smooth_alpha)
    render_smoother = RenderSmoother(follow=0.6)
    cv_tracker = CVFaceTracker()
    use_cv_tracker = trackers_available()
    if use_cv_tracker:
        print("[Pipeline] ใช้ OpenCV tracker (KCF) — กรอบตามหน้าทุกเฟรม")
    else:
        print("[Pipeline] opencv ไม่มี tracker — ใช้ RenderSmoother แทน "
              "(ลง opencv-contrib-python เพื่อเปิด tracker)")
    broadcaster = ws_broadcaster
    Path(s.snapshots_dir).mkdir(parents=True, exist_ok=True)


def _save_face(frame, x, y, w, h, name: str) -> str:
    s = config.s
    folder = Path(s.snapshots_dir) / name
    folder.mkdir(parents=True, exist_ok=True)

    pad = 30
    y1, y2 = max(y - pad, 0), min(y + h + pad, frame.shape[0])
    x1, x2 = max(x - pad, 0), min(x + w + pad, frame.shape[1])
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return ""

    if engine.is_duplicate(face_crop, folder, s.dedup_thresh):
        return ""

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(folder / f"{name}_{ts}.jpg")
    cv2.imwrite(path, face_crop)
    return path


def detection_worker():
    """ตรวจจับ+ระบุตัวตน ใน background thread"""
    print("[Worker] เริ่มทำงาน...")
    s = config.s
    rebuild_counter = 0

    while state.running:
        frame, seq = state.get_frame_with_seq()
        if frame is None or not DEEPFACE_AVAILABLE:
            time.sleep(0.05)
            continue

        # ข้ามถ้าเป็นเฟรมเดิมที่ detect ไปแล้ว (กัน worker วนเปล่ากิน CPU แย่งกับการแสดงผล)
        if seq == _last_frame_seq[0]:
            time.sleep(0.008)
            continue
        _last_frame_seq[0] = seq

        # เช็คว่ามีการเพิ่ม/ลบ known face ไหม (ทุกๆ ~50 รอบ ลดภาระ)
        rebuild_counter += 1
        if rebuild_counter >= 50:
            rebuild_counter = 0
            engine.maybe_rebuild()

        # อัปเดตพารามิเตอร์ที่แก้ได้ runtime
        tracker.leave_timeout = s.leave_timeout
        smoother.alpha = s.smooth_alpha

        scale = s.scale
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        new_results = []

        try:
            # ---------- ชั้นที่ 1: หาตำแหน่งหน้า (เร็ว) ----------
            # ใช้ mediapipe ถ้ามี (2-5ms) ไม่มีก็ fallback DeepFace detector
            if fast_detector is not None and fast_detector.available:
                dets = fast_detector.detect(small)
                faces = [{"confidence": d["confidence"],
                          "facial_area": {"x": d["box"][0], "y": d["box"][1],
                                          "w": d["box"][2], "h": d["box"][3]}}
                         for d in dets]
                conf_needed = 0.6      # เกณฑ์ของ mediapipe (สเกลต่างจาก DeepFace)
            else:
                faces = engine.detect_faces(small)
                conf_needed = s.min_confidence

            tracker.cleanup()
            smoother.cleanup(timeout=s.leave_timeout + 2)
            state.expire_in_view(s.leave_timeout)

            # ---------- ชั้นที่ 2: ระบุตัวตน (หนัก ทำเป็นช่วง) ----------
            # identify ทุก IDENTIFY_EVERY รอบ detect เท่านั้น
            # ระหว่างนั้นใช้ชื่อเดิมจากตำแหน่งใกล้เคียง (กรอบยังลื่น ชื่อตามหลังนิดหน่อย)
            _identify_counter[0] += 1
            do_identify = (_identify_counter[0] % IDENTIFY_EVERY == 0) or not _name_cache

            for face_obj in faces:
                if face_obj.get("confidence", 0) < conf_needed:
                    continue
                area = face_obj.get("facial_area", {})
                rx = int(area.get("x", 0) / scale)
                ry = int(area.get("y", 0) / scale)
                rw = int(area.get("w", 0) / scale)
                rh = int(area.get("h", 0) / scale)
                if rw < s.min_face_px or rh < s.min_face_px:
                    continue

                cx, cy = rx + rw // 2, ry + rh // 2

                if do_identify:
                    # crop หน้าจากเฟรมเต็ม แล้วให้ DeepFace ระบุชื่อ
                    pad = 20
                    fy1, fy2 = max(ry - pad, 0), min(ry + rh + pad, frame.shape[0])
                    fx1, fx2 = max(rx - pad, 0), min(rx + rw + pad, frame.shape[1])
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    if face_crop.size == 0:
                        continue
                    name, _dist = engine.identify(face_crop, s.distance_thresh)
                    _name_cache[:] = [c for c in _name_cache
                                      if abs(c[0] - cx) > rw or abs(c[1] - cy) > rh]
                    _name_cache.append((cx, cy, name))
                else:
                    # ใช้ชื่อเดิมจากตำแหน่งที่ใกล้ที่สุด (ไม่เรียก DeepFace = ลื่น)
                    name = "UNKNOWN"
                    best = None
                    for (px, py, pname) in _name_cache:
                        d = abs(px - cx) + abs(py - cy)
                        if best is None or d < best[0]:
                            best = (d, pname)
                    if best is not None and best[0] < max(rw, rh) * 2:
                        name = best[1]

                sx, sy, sw, sh = smoother.update(name, (rx, ry, rw, rh))
                new_results.append({"box": (sx, sy, sw, sh), "name": name})
                state.mark_in_view(name)

                # ตัดสินใจ snap/log — ทำเฉพาะรอบที่ identify จริง (ชื่อเชื่อถือได้)
                if not do_identify:
                    continue
                should_snap = (name != "UNKNOWN") or s.snap_unknown
                if not should_snap or tracker.is_active(name):
                    tracker.seen(name)
                    continue

                tracker.seen(name)
                snap_path = _save_face(frame, rx, ry, rw, rh, name)
                if snap_path:
                    status = "known" if name != "UNKNOWN" else "unknown"
                    entry = event_logger.log(name, snap_path, status)
                    if broadcaster:
                        broadcaster.broadcast_threadsafe({"type": "new_detection", "entry": entry})

                # ---- บันทึกเวลาเข้า/ออกงาน ----
                if name != "UNKNOWN":
                    att = attendance.mark_seen(name, snap_path)
                    if att and broadcaster:
                        broadcaster.broadcast_threadsafe({"type": "attendance", "entry": att})
                else:
                    # คนแปลกหน้า เก็บแยก (แต่ไม่รัวเกินไป — snap_path มีค่าแปลว่าไม่ใช่ภาพซ้ำ)
                    if snap_path:
                        attendance.log_stranger(snap_path)

        except Exception as e:
            print(f"[Worker] error: {e}")

        state.set_results(new_results)
        render_smoother.set_targets(new_results)   # ตั้งเป้าหมายให้กรอบไหลตาม (fallback)
        render_smoother.cleanup()
        # ตั้งต้น cv tracker จากผล detect ที่ยืนยันแล้ว (ตามหน้าทุกเฟรมใน capture loop)
        if use_cv_tracker:
            cv_tracker.reset_tracks(frame, new_results)

        if broadcaster:
            broadcaster.broadcast_threadsafe(
                {"type": "stats_update", "in_view": state.in_view_count()}
            )
        time.sleep(0.01)


def capture_loop():
    """อ่านกล้องตลอดเวลา + วาดกรอบ + เก็บ annotated frame"""
    RETRY_DELAY = 3.0

    while state.running:
        active_source = config.s.camera_source
        cap = open_camera(active_source)
        if cap is None or not cap.isOpened():
            print(f"[ERROR] เปิดกล้องไม่ได้: {active_source} — ลองใหม่ใน {RETRY_DELAY} วิ")
            state.camera_connected = False
            wait_start = time.time()
            while state.running and time.time() - wait_start < RETRY_DELAY:
                if config.s.camera_source != active_source:
                    break
                time.sleep(0.1)
            continue

        state.camera_connected = True
        fps_timer = time.time()
        frame_cnt = 0
        tick = 0
        print(f"[Camera] เริ่มดึงภาพจาก: {active_source}")

        while state.running:
            if config.s.camera_source != active_source:
                print(f"[Camera] เปลี่ยนแหล่งเป็น {config.s.camera_source} → เชื่อมต่อใหม่")
                break

            ret, frame = cap.read()
            if not ret:
                print(f"[Camera] กล้องหลุด — ลองเชื่อมต่อใหม่ใน {RETRY_DELAY} วิ")
                state.camera_connected = False
                break

            tick += 1
            frame_cnt += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                state.current_fps = frame_cnt / elapsed
                fps_timer = time.time()
                frame_cnt = 0

            if tick % config.s.process_every == 0:
                state.set_frame(frame.copy())

            # กรอบตามหน้าทุกเฟรม: ใช้ cv tracker (ตามจากภาพจริง) ถ้ามี
            # ไม่มีก็ fallback เป็น RenderSmoother (interpolate เข้าหาเป้าหมาย)
            if use_cv_tracker and cv_tracker is not None:
                boxes = cv_tracker.update(frame)
            elif render_smoother:
                boxes = render_smoother.step()
            else:
                boxes = state.get_results()
            for r in boxes:
                x, y, w, h = r["box"]
                drawing.draw_box(frame, x, y, w, h, r["name"], drawing.color_for(r["name"]))
            drawing.draw_hud(frame, state.current_fps, len(boxes))

            # ส่งเฟรมนี้ให้สตรีมโดยตรง (frame ถูกใช้เสร็จแล้วในรอบนี้ ไม่ต้อง copy)
            state.set_annotated(frame)
            # ไม่ sleep — ปล่อยให้เร็วตามกล้อง cap.read() จะบล็อกเองตาม fps กล้อง

        cap.release()
        if state.running and config.s.camera_source == active_source:
            time.sleep(RETRY_DELAY)

    state.camera_connected = False

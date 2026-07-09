"""
============================================================
  AI Face Verification — FastAPI Backend (refactored)
============================================================
โครงสร้างใหม่:
  app/core/config.py      → จัดการ config รวมศูนย์ (thread-safe)
  app/core/state.py       → shared state ระหว่าง thread
  app/services/
      face_engine.py      → embedding cache + cosine (แก้ความหน่วง)
      camera.py           → เปิดกล้อง USB/RTSP เสถียร
      pipeline.py         → capture loop + detection worker
      tracking.py         → tracker + box smoother
      logger.py           → บันทึกเหตุการณ์
      drawing.py          → วาดกรอบ/HUD
      websocket.py        → real-time push
  app/api/routes.py       → API endpoints ทั้งหมด

รัน:
  uvicorn backend:app --host 0.0.0.0 --port 8000
  หรือ  python backend.py  (หาพอร์ตว่าง + เปิดเบราว์เซอร์ให้)

ติดตั้ง:
  pip install fastapi "uvicorn[standard]" opencv-python deepface tf-keras numpy python-multipart
============================================================
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import BASE_DIR
from app.core.state import state
from app.api.routes import router
from app.services import pipeline
from app.services.websocket import manager

app = FastAPI(title="AI Face Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup():
    manager.loop = asyncio.get_event_loop()
    # ประกอบ pipeline (โหลดโมเดล + สร้าง embedding cache ครั้งเดียว)
    pipeline.init_pipeline(ws_broadcaster=manager)
    threading.Thread(target=pipeline.capture_loop, daemon=True).start()
    threading.Thread(target=pipeline.detection_worker, daemon=True).start()
    print("[System] เริ่มระบบเรียบร้อย")


@app.on_event("shutdown")
async def shutdown():
    state.running = False


# ---------- STATIC FRONTEND (mount ท้ายสุด) ----------
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import socket
    import time
    import webbrowser
    import uvicorn

    PORT = 8000  # พอร์ตคงที่ เพื่อให้คนอื่นเข้าถึงด้วย URL เดิมได้ตลอด

    # หา IP ของเครื่องนี้ในวง LAN เพื่อบอกให้คนอื่นเข้าถึง
    def get_lan_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))   # ไม่ได้ส่งจริง แค่ให้ OS เลือก interface
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    lan_ip = get_lan_ip()

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    print("\n" + "=" * 52)
    print("  ระบบพร้อมใช้งาน — แชร์ลิงก์ให้คนในออฟฟิศ:")
    print(f"    เครื่องนี้ (server) : http://127.0.0.1:{PORT}")
    print(f"    เครื่องอื่นในวง LAN : http://{lan_ip}:{PORT}")
    print("=" * 52 + "\n")

    threading.Thread(target=open_browser, daemon=True).start()
    # host 0.0.0.0 = เปิดให้เครื่องอื่นในวงเน็ตเวิร์กเดียวกันเข้าถึงได้
    uvicorn.run("backend:app", host="0.0.0.0", port=PORT, log_level="info")

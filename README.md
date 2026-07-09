# AI Face Verification — โครงสร้างใหม่

## รัน
```bash
pip install -r requirements.txt
python backend.py            # หาพอร์ตว่าง + เปิดเบราว์เซอร์ให้อัตโนมัติ
# หรือ
uvicorn backend:app --host 0.0.0.0 --port 8000
```

## โครงสร้าง
```
backend.py                  entry point
config.json                 (สร้างอัตโนมัติเมื่อแก้ค่าผ่าน API)
app/
  core/
    config.py               config รวมศูนย์ thread-safe (dataclass)
    state.py                shared state + lock ทุกตัว
  services/
    face_engine.py          embedding cache + cosine  ← แก้ความหน่วง
    camera.py               เปิดกล้อง USB/RTSP เสถียร
    pipeline.py             capture loop + detection worker
    tracking.py             tracker + box smoother
    logger.py               บันทึกเหตุการณ์
    drawing.py              วาดกรอบ/HUD
    websocket.py            real-time push
  api/
    routes.py               API endpoints ทั้งหมด
static/                     (วางไฟล์หน้าเว็บเดิมไว้ที่นี่)
known_faces/                รูปคนที่ลงทะเบียน (โฟลเดอร์ต่อคน)
snapshots/                  ภาพที่ระบบ capture
```

## สิ่งที่แก้จากเดิม
1. **ความหน่วง/ไม่ realtime**: เลิกใช้ `DeepFace.verify()` วนทีละไฟล์
   เปลี่ยนเป็น cache embedding ของ known faces ทั้งหมดครั้งเดียว
   แล้วเทียบ cosine ทั้งก้อนด้วย numpy — เทียบ 100+ คนใช้เวลา ~0.004ms
   (rebuild cache อัตโนมัติเมื่อเพิ่ม/ลบคน)
2. **กล้องเสถียรขึ้น**: แยก USB/RTSP ชัดเจน, warm-up ยืดหยุ่น,
   RTSP ตั้ง buffer=1 ลด latency, `force_mjpg` ปิดได้ถ้ากล้องไม่รองรับ
3. **โครงสร้างเป็นระบบ**: แยก config / state / services / api
   ออกจากกัน แทนไฟล์เดียว 766 บรรทัด

## endpoint สำคัญ
- `GET  /video_feed`            สตรีม MJPEG
- `GET  /api/status`           สถานะกล้อง/fps/คนหน้ากล้อง
- `GET  /api/logs`             ดึง log ตามช่วงเวลา
- `POST /api/register`         ลงทะเบียนใบหน้า (rebuild cache ทันที)
- `GET/POST /api/config`       ดู/แก้ config
- `WS   /ws`                   real-time push

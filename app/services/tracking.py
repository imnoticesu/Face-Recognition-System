"""
Tracking utilities — dedup ตอนคนยังอยู่หน้ากล้อง + ทำให้กรอบนิ่ง
"""
from __future__ import annotations

import time


class FaceTracker:
    """กันการ snap/log ซ้ำ ระหว่างที่คนคนเดิมยังอยู่หน้ากล้อง"""

    def __init__(self, leave_timeout: float = 5.0):
        self.active: dict[str, float] = {}
        self.leave_timeout = leave_timeout

    def is_active(self, name: str) -> bool:
        return name in self.active

    def seen(self, name: str):
        self.active[name] = time.time()

    def cleanup(self):
        now = time.time()
        for n in [n for n, t in self.active.items() if now - t > self.leave_timeout]:
            del self.active[n]


class BoxSmoother:
    """EMA smoothing ให้กรอบไม่กระตุก"""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.tracks: dict[str, dict] = {}

    def update(self, name: str, box: tuple) -> tuple:
        x, y, w, h = box
        if name not in self.tracks:
            self.tracks[name] = {"box": [float(x), float(y), float(w), float(h)],
                                 "last_seen": time.time()}
            return box
        prev = self.tracks[name]["box"]
        a = self.alpha
        sx = a * x + (1 - a) * prev[0]
        sy = a * y + (1 - a) * prev[1]
        sw = a * w + (1 - a) * prev[2]
        sh = a * h + (1 - a) * prev[3]
        self.tracks[name]["box"] = [sx, sy, sw, sh]
        self.tracks[name]["last_seen"] = time.time()
        return (int(sx), int(sy), int(sw), int(sh))

    def cleanup(self, timeout: float = 6.0):
        now = time.time()
        for n in [n for n, v in self.tracks.items() if now - v["last_seen"] > timeout]:
            del self.tracks[n]


class RenderSmoother:
    """
    ทำให้กรอบขยับนุ่มในทุกเฟรมของกล้อง (ไม่ใช่แค่ตอน worker อัปเดต)
    worker ตั้ง 'เป้าหมาย' (target) เป็นระยะ ๆ ตามที่ detect ได้
    ส่วน capture loop เรียก step() ทุกเฟรมเพื่อเลื่อน current เข้าหา target ทีละนิด
    → เห็นกรอบไหลลื่นแม้ worker ช้ากว่ากล้องมาก
    """
    def __init__(self, follow: float = 0.35):
        self.follow = follow                 # 0..1 ยิ่งสูงตามไว ยิ่งต่ำนุ่ม
        self.targets: dict[str, list] = {}   # เป้าหมายล่าสุดจาก worker
        self.current: dict[str, list] = {}   # ตำแหน่งที่กำลังวาดจริง
        self.last_seen: dict[str, float] = {}

    def set_targets(self, results: list[dict]):
        """เรียกจาก worker: อัปเดตเป้าหมายของแต่ละคน"""
        now = time.time()
        seen = set()
        for r in results:
            key = r["name"] + "_" + str(r.get("id", ""))  # แยกตามชื่อ (พอสำหรับ use case นี้)
            key = r["name"]
            seen.add(key)
            self.targets[key] = [float(v) for v in r["box"]] + [r["name"]]
            self.last_seen[key] = now
            if key not in self.current:
                self.current[key] = list(self.targets[key])

    def step(self) -> list[dict]:
        """เรียกทุกเฟรมกล้อง: เลื่อน current เข้าหา target แล้วคืนกรอบที่จะวาด

        ใช้ adaptive follow: ถ้าหน้าขยับไกล (ขยับเร็ว) จะไล่ตามเร็วขึ้นอัตโนมัติ
        ทำให้ตามทันตอนขยับไว แต่ยังนุ่มตอนอยู่นิ่ง
        """
        out = []
        for key, tgt in list(self.targets.items()):
            cur = self.current.get(key, list(tgt))
            # ระยะที่ต้องขยับ (ใช้แกน x,y ประเมินความเร็วการเคลื่อนที่)
            dist = abs(tgt[0] - cur[0]) + abs(tgt[1] - cur[1])
            # ขยับไกล → follow เข้าใกล้ 0.9 (ไล่ตามไว), ขยับนิด → follow ต่ำ (นุ่ม)
            f = min(0.9, self.follow + dist / 200.0)
            for i in range(4):
                cur[i] = f * tgt[i] + (1 - f) * cur[i]
            cur[4] = tgt[4]  # ชื่อ
            self.current[key] = cur
            out.append({"box": (int(cur[0]), int(cur[1]), int(cur[2]), int(cur[3])),
                        "name": cur[4]})
        return out

    def cleanup(self, timeout: float = 1.5):
        now = time.time()
        for k in [k for k, t in self.last_seen.items() if now - t > timeout]:
            self.targets.pop(k, None)
            self.current.pop(k, None)
            self.last_seen.pop(k, None)

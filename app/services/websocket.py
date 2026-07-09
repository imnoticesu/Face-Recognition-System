"""
WebSocket connection manager — push เหตุการณ์ไปหน้าเว็บแบบ real-time
"""
from __future__ import annotations

import asyncio
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def _broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_threadsafe(self, message: dict):
        """เรียกจาก worker thread (sync) เพื่อ push แบบ thread-safe"""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)


manager = ConnectionManager()

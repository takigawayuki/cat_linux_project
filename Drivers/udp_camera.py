"""
UDP frame receiver for FPGA camera (320x240 RGB565, port 1234).
Packet protocol:
  - pkt 1~240 : row data (pkt==1 has 8-byte header before pixels)
  - pkt 241   : ROI packet (8 bytes: x1,y1,x2,y2 as uint16 big-endian)
"""
import time
import socket
import struct
import threading
import numpy as np

PORT          = 1234
WIDTH         = 320
HEIGHT        = 240
TOTAL_PACKETS = 241


class UDPCamera:
    def __init__(self, port=PORT):
        self._raw     = [np.zeros((HEIGHT, WIDTH), dtype=np.uint16) for _ in range(2)]
        self._write   = 0
        self._frame   = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self._roi     = (0, 0, WIDTH, HEIGHT)
        self._lock    = threading.Lock()
        self._event   = threading.Event()
        self.cap_fps  = 0.0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, 41, 64 * 1024 * 1024)  # SO_RCVBUFFORCE
        except Exception:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self._sock.bind(('', port))
        print(f'UDPCamera listening on port {port}')

        threading.Thread(target=self._recv_loop, daemon=True).start()

    def _recv_loop(self):
        received      = [False] * (HEIGHT + 2)
        rows_received = 0
        last_time     = time.time()

        while True:
            data, _ = self._sock.recvfrom(2048)
            if len(data) < 4:
                continue

            pkt = struct.unpack('!I', data[:4])[0]
            if pkt == 0 or pkt > TOTAL_PACKETS:
                continue

            payload = data[4:]

            # ROI packet
            if pkt == TOTAL_PACKETS:
                if len(payload) >= 8:
                    x1, y1, x2, y2 = struct.unpack('!HHHH', payload[:8])
                    self._roi = (x1, y1, x2, y2)
                continue

            row = pkt - 1
            if row < 0 or row >= HEIGHT:
                continue

            if pkt == 1:
                if len(payload) < 8 + WIDTH * 2:
                    continue
                pixels = np.frombuffer(payload[8:8 + WIDTH * 2], dtype='>u2')
            else:
                if len(payload) < WIDTH * 2:
                    continue
                pixels = np.frombuffer(payload[:WIDTH * 2], dtype='>u2')

            if not received[pkt]:
                received[pkt] = True
                rows_received += 1

            self._raw[self._write][row] = pixels

            if rows_received == HEIGHT:
                # RGB565 → BGR888
                src = self._raw[self._write]
                r = ((src >> 11) & 0x1F).astype(np.uint8)
                g = ((src >> 5)  & 0x3F).astype(np.uint8)
                b = (src         & 0x1F).astype(np.uint8)
                bgr = np.stack([
                    (b << 3) | (b >> 2),
                    (g << 2) | (g >> 4),
                    (r << 3) | (r >> 2),
                ], axis=2).astype(np.uint8)

                with self._lock:
                    self._frame = bgr
                    self._write = 1 - self._write

                now          = time.time()
                self.cap_fps = 1.0 / max(now - last_time, 1e-6)
                last_time    = now

                self._event.set()
                received      = [False] * (HEIGHT + 2)
                rows_received = 0

    def read(self):
        """Block until next frame. Returns (bgr_frame, roi, cap_fps).
        roi = (x1, y1, x2, y2) in pixel coords, from FPGA ROI packet.
        """
        self._event.wait()
        self._event.clear()
        with self._lock:
            return self._frame.copy(), self._roi, self.cap_fps

    def release(self):
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

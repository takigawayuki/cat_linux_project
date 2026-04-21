#!/usr/bin/env python3
import socket
import struct
import numpy as np
import cv2
import threading
import queue

LOCAL_IP = '0.0.0.0'
PORT = 1234

WIDTH = 720
HEIGHT = 480

FRAME_HEADER = 0xf05aa50f

# 🚀 队列（关键）
packet_queue = queue.Queue(maxsize=2000)


class UDPReceiver:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((LOCAL_IP, PORT))

        self.socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            16 * 1024 * 1024
        )

        print(f"监听 {LOCAL_IP}:{PORT}")

    def run(self):
        while True:
            data, addr = self.socket.recvfrom(2048)

            print(f"收到UDP包: {len(data)} bytes 来自 {addr}")  # 👈关键

            try:
                packet_queue.put_nowait(data)
            except queue.Full:
                print("⚠️ 队列满，丢包")


class ImageProcessor:
    def __init__(self):
        self.image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.received_packets = set()

        self.x1 = self.y1 = self.x2 = self.y2 = 0
        self.frame_count = 0

    def reset_frame(self):
        self.received_packets.clear()
        self.image[:] = 0

    def process_packet(self, data):
        if len(data) < 4:
            return

        packet_num = struct.unpack('!I', data[:4])[0]
        payload = data[4:]

        # ========= 帧头 =========
        if len(payload) >= 8:
            header = struct.unpack('!I', payload[:4])[0]
            if header == FRAME_HEADER:
                self.reset_frame()
                payload = payload[8:]

        # ========= 最后一包 =========
        if len(payload) == 8 or len(payload) == 16:
            if len(payload) >= 8:
                self.x1, self.y1, self.x2, self.y2 = struct.unpack('!HHHH', payload[:8])

            total_expected = packet_num - 1

            if len(self.received_packets) >= total_expected:
                self.frame_count += 1
                print(f"Frame {self.frame_count} OK")
                self.show_image()

            return

        # ========= 图像 =========
        if packet_num in self.received_packets:
            return

        self.received_packets.add(packet_num)

        row = (packet_num - 1) // 2
        col_start = ((packet_num - 1) % 2) * (WIDTH // 2)

        if len(payload) % 2 != 0:
            return

        pixels = np.frombuffer(payload, dtype='>u2')

        r = ((pixels >> 11) & 0x1F).astype(np.uint8) << 3
        g = ((pixels >> 5) & 0x3F).astype(np.uint8) << 2
        b = (pixels & 0x1F).astype(np.uint8) << 3

        line = np.stack((b, g, r), axis=1)

        end = col_start + len(line)

        if row < HEIGHT and end <= WIDTH:
            self.image[row, col_start:end] = line

    def show_image(self):
        x1 = min(self.x1, self.x2)
        x2 = max(self.x1, self.x2)
        y1 = min(self.y1, self.y2)
        y2 = max(self.y1, self.y2)

        x1 = max(0, min(x1, WIDTH - 1))
        x2 = max(0, min(x2, WIDTH - 1))
        y1 = max(0, min(y1, HEIGHT - 1))
        y2 = max(0, min(y2, HEIGHT - 1))

        if x2 > x1 and y2 > y1:
            crop = self.image[y1:y2, x1:x2]
            crop = cv2.resize(crop, (WIDTH, HEIGHT))
            cv2.imshow("FPGA", crop)
        else:
            cv2.imshow("FPGA", self.image)

        cv2.waitKey(1)

    def run(self):
        print("处理线程启动")  # 👈关键

        while True:
            data = packet_queue.get()
            print(f"处理一个包: {len(data)} bytes")  # 👈关键
            self.process_packet(data)


if __name__ == "__main__":
    receiver = UDPReceiver()
    processor = ImageProcessor()

    t1 = threading.Thread(target=receiver.run, daemon=True)
    t2 = threading.Thread(target=processor.run, daemon=True)

    t1.start()
    t2.start()

    t1.join()
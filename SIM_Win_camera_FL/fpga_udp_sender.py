import socket
import struct
import time
import cv2
import numpy as np

TARGET_IP   = '192.168.137.5'
TARGET_PORT = 1234
IMG_WIDTH   = 640
IMG_HEIGHT  = 480
FRAME_HEAD  = 0xF05AA50F
FPS         = 60

ROI_X1, ROI_Y1 = 160, 120
ROI_X2, ROI_Y2 = 480, 360

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

def frame_to_rgb565(frame):
    """BGR frame → RGB565 big-endian, shape (HEIGHT, WIDTH) uint16"""
    b = frame[:, :, 0].astype(np.uint16)
    g = frame[:, :, 1].astype(np.uint16)
    r = frame[:, :, 2].astype(np.uint16)
    rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    # 转大端字节序
    return ((rgb565 >> 8) | (rgb565 << 8)).astype(np.uint16)

def send_frame(frame):
    rgb565 = frame_to_rgb565(frame)  # (480, 640) uint16, big-endian

    # 包1: pkt_num(4) + frame_head(4) + W(2) + H(2) + row0像素
    hdr = struct.pack('>IIHH', 1, FRAME_HEAD, IMG_WIDTH, IMG_HEIGHT)
    sock.sendto(hdr + rgb565[0].tobytes(), (TARGET_IP, TARGET_PORT))

    # 包2-480
    for row in range(1, IMG_HEIGHT):
        pkt_hdr = struct.pack('>I', row + 1)
        sock.sendto(pkt_hdr + rgb565[row].tobytes(), (TARGET_IP, TARGET_PORT))

    # 包481: ROI
    sock.sendto(
        struct.pack('>IHHHH', 481, ROI_X1, ROI_Y1, ROI_X2, ROI_Y2),
        (TARGET_IP, TARGET_PORT)
    )

def main():
    frame_interval = 1.0 / FPS
    frame_count = 0
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("capture failed")
                continue
            send_frame(frame)
            frame_count += 1
            print(f"\rFrame {frame_count}", end='', flush=True)
            elapsed = time.perf_counter() - t0
            wait = frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        sock.close()

if __name__ == "__main__":
    main()

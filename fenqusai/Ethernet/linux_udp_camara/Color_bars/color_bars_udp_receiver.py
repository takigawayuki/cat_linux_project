#!/usr/bin/env python3
"""
Receive 640x360 RGB565 color-bar frames from FPGA over UDP and display them.

FPGA UDP payload protocol:
  Packet 1:
    uint32 packet_num     big-endian, value 1
    uint32 frame_header   big-endian, value 0xf05aa50f
    uint16 width          big-endian, value 640
    uint16 height         big-endian, value 360
    uint16 pixels[640]    big-endian RGB565, first image row

  Packet 2..360:
    uint32 packet_num     big-endian, value 2..360
    uint16 pixels[640]    big-endian RGB565, one image row

One complete frame has 360 UDP packets. packet_num - 1 is the row index.
"""

import argparse
import socket
import time

import cv2
import numpy as np


LOCAL_IP = "0.0.0.0"
PORT = 1234
FPGA_IP = "192.168.1.11"

WIDTH = 640
HEIGHT = 360
TOTAL_PACKETS = HEIGHT
FRAME_HEADER = 0xF05AA50F

ROW_BYTES = WIDTH * 2
FIRST_PACKET_BYTES = 4 + 4 + 4 + ROW_BYTES
ROW_PACKET_BYTES = 4 + ROW_BYTES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Receive FPGA UDP color bars: 640x360 RGB565 -> BGR888."
    )
    parser.add_argument("--local-ip", default=LOCAL_IP,
                        help="Local IP to bind. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=PORT,
                        help="UDP port. Default: 1234")
    parser.add_argument("--fpga-ip", default=FPGA_IP,
                        help="Only accept packets from this FPGA IP. Use empty string to disable.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Do not filter packets by source IP.")
    parser.add_argument("--stats-interval", type=float, default=1.0,
                        help="Seconds between status prints. Default: 1.0")
    parser.add_argument("--no-display", action="store_true",
                        help="Receive and print stats without cv2.imshow.")
    return parser.parse_args()


def u32_be(buf):
    return int.from_bytes(buf, "big")


def u16_be(buf):
    return int.from_bytes(buf, "big")


def rgb565_to_bgr888(raw565):
    r5 = ((raw565 >> 11) & 0x1F).astype(np.uint8)
    g6 = ((raw565 >> 5) & 0x3F).astype(np.uint8)
    b5 = (raw565 & 0x1F).astype(np.uint8)

    b = (b5 << 3) | (b5 >> 2)
    g = (g6 << 2) | (g6 >> 4)
    r = (r5 << 3) | (r5 >> 2)

    return np.dstack((b, g, r)).astype(np.uint8)


def make_socket(local_ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
    except OSError:
        pass
    sock.bind((local_ip, port))

    actual_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"Listening on {local_ip}:{port}, SO_RCVBUF={actual_buf // 1024} KB")
    return sock


def main():
    args = parse_args()
    filter_ip = "" if args.no_filter else args.fpga_ip

    sock = make_socket(args.local_ip, args.port)

    raw = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    received = np.zeros(HEIGHT, dtype=bool)
    rows_received = 0
    synced = False

    frames_ok = 0
    packets_seen = 0
    packets_dropped = 0
    bad_packets = 0
    last_frame_time = time.time()
    last_stats_time = time.time()

    if filter_ip:
        print(f"Accepting packets only from FPGA IP {filter_ip}")
    else:
        print("Source-IP filtering disabled")

    print("Expected protocol: port=1234, 640x360, 360 packets/frame, RGB565 big-endian")
    print("Press q or ESC in the image window to quit.")

    while True:
        data, addr = sock.recvfrom(4096)
        src_ip, src_port = addr

        if filter_ip and src_ip != filter_ip:
            continue

        packets_seen += 1

        if len(data) < 4:
            bad_packets += 1
            continue

        pkt = u32_be(data[0:4])
        if pkt < 1 or pkt > TOTAL_PACKETS:
            bad_packets += 1
            if bad_packets <= 5:
                print(f"Bad packet number {pkt}, len={len(data)}, head={data[:16].hex(' ')}")
            continue

        if pkt == 1:
            if len(data) < FIRST_PACKET_BYTES:
                bad_packets += 1
                print(f"Short first packet: len={len(data)}, expected>={FIRST_PACKET_BYTES}")
                continue

            frame_header = u32_be(data[4:8])
            width = u16_be(data[8:10])
            height = u16_be(data[10:12])

            if frame_header != FRAME_HEADER or width != WIDTH or height != HEIGHT:
                bad_packets += 1
                print(
                    "First packet header mismatch: "
                    f"frame_header=0x{frame_header:08x}, width={width}, height={height}, "
                    f"len={len(data)}, head={data[:24].hex(' ')}"
                )
                continue

            if synced and rows_received != HEIGHT:
                packets_dropped += HEIGHT - rows_received

            received[:] = False
            rows_received = 0
            synced = True

            pixels = np.frombuffer(data[12:12 + ROW_BYTES], dtype=">u2")
            raw[0, :] = pixels
            received[0] = True
            rows_received = 1

        else:
            if not synced:
                continue

            if len(data) < ROW_PACKET_BYTES:
                bad_packets += 1
                if bad_packets <= 5:
                    print(f"Short row packet: pkt={pkt}, len={len(data)}, expected>={ROW_PACKET_BYTES}")
                continue

            row = pkt - 1
            pixels = np.frombuffer(data[4:4 + ROW_BYTES], dtype=">u2")
            raw[row, :] = pixels

            if not received[row]:
                received[row] = True
                rows_received += 1

        if rows_received == HEIGHT:
            frame_bgr = rgb565_to_bgr888(raw)
            frames_ok += 1

            now = time.time()
            fps = 1.0 / max(now - last_frame_time, 1e-6)
            last_frame_time = now

            if not args.no_display:
                cv2.imshow("FPGA Color Bars 640x360", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if now - last_stats_time >= args.stats_interval:
                print(
                    f"frames={frames_ok} fps={fps:.1f} "
                    f"packets={packets_seen} bad={bad_packets} "
                    f"partial_lost_rows={packets_dropped} src={src_ip}:{src_port}"
                )
                last_stats_time = now

            synced = False
            received[:] = False
            rows_received = 0

    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Display FPGA UDP image stream using the shared Drivers.fpga_udp_camera interface.

This script is only a small runnable demo for the current 640x360 stream.
Packet parsing, row-based frame assembly, and RGB565 -> BGR888 conversion live in
Drivers/fpga_udp_camera.py.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Drivers.fpga_udp_camera import FPGAUDPCamera, FPGAUDPImageProtocol, FPGA_IP, PORT  # noqa: E402


PROTOCOL = FPGAUDPImageProtocol(width=640, height=360)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Display FPGA UDP stream: 640x360 RGB565 -> BGR888."
    )
    parser.add_argument("--local-ip", default="",
                        help="Local IP to bind. Default: all interfaces.")
    parser.add_argument("--port", type=int, default=PORT,
                        help="UDP port. Default: 1234")
    parser.add_argument("--fpga-ip", default=FPGA_IP,
                        help="Only accept packets from this FPGA IP. Default: 192.168.1.11")
    parser.add_argument("--no-filter", action="store_true",
                        help="Do not filter packets by source IP.")
    parser.add_argument("--stats-interval", type=float, default=1.0,
                        help="Seconds between status prints. Default: 1.0")
    parser.add_argument("--no-display", action="store_true",
                        help="Receive and print stats without cv2.imshow.")
    return parser.parse_args()


def main():
    args = parse_args()
    cam = FPGAUDPCamera(
        port=args.port,
        local_ip=args.local_ip,
        fpga_ip=args.fpga_ip,
        filter_source=not args.no_filter,
        protocol=PROTOCOL,
    )

    print("Expected protocol: port=1234, 640x360, 360 packets/frame, RGB565 big-endian")
    print("Press q or ESC in the image window to quit.")

    last_stats_time = time.time()

    try:
        while True:
            result = cam.read_latest(timeout=0.5)
            if result is None:
                continue

            frame_bgr, _, cap_fps = result

            if not args.no_display:
                cv2.imshow("FPGA UDP Camera 640x360", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            now = time.time()
            if now - last_stats_time >= args.stats_interval:
                stats = cam.stats()
                print(
                    f"frames={stats['frames_ok']} fps={cap_fps:.1f} "
                    f"packets={stats['packets_seen']} bad={stats['bad_packets']} "
                    f"partial_lost_rows={stats['partial_lost_rows']} "
                    f"last_error={stats['last_error'] or '-'}"
                )
                last_stats_time = now
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

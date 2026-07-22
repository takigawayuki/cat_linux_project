#!/usr/bin/env python3
"""
High-throughput viewer for the FPGA UDP camera stream.

Thread model:
  - FPGAUDPCamera internal receiver thread:
      recvfrom_into() -> validate packet -> assemble raw RGB565 rows into 3 buffers.
      It does not display and does not convert RGB565 to BGR888.
  - Main/display side:
      always reads the latest complete raw frame, drops older complete frames,
      converts RGB565 -> BGR888, and displays with OpenCV.

This file is for the real FPGA camera image stream. The transport protocol is the
current actual-image stream: 480x270, 270 row packets, UDP/1234,
big-endian RGB565.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Drivers.fpga_udp_camera import (  # noqa: E402
    FPGAUDPCamera,
    FPGAUDPImageProtocol,
    FPGA_IP,
    PORT,
)


PROTOCOL = FPGAUDPImageProtocol(width=480, height=270)
WIDTH = PROTOCOL.width
HEIGHT = PROTOCOL.height
WINDOW_NAME = 'FPGA UDP Camera 480x270'


def parse_args():
    parser = argparse.ArgumentParser(
        description='High-throughput FPGA UDP camera viewer.'
    )
    parser.add_argument('--local-ip', default='',
                        help='Local IP to bind. Default: all interfaces.')
    parser.add_argument('--port', type=int, default=PORT,
                        help='UDP port. Default: 1234.')
    parser.add_argument('--fpga-ip', default=FPGA_IP,
                        help='Only accept packets from this FPGA IP. Default: 192.168.1.11.')
    parser.add_argument('--no-filter', action='store_true',
                        help='Do not filter packets by source IP.')
    parser.add_argument('--recv-buf-mb', type=int, default=64,
                        help='Socket receive buffer request in MB. Default: 64.')
    parser.add_argument('--stats-interval', type=float, default=3.0,
                        help='Seconds between stats prints. Default: 3.0.')
    parser.add_argument('--wait-timeout', type=float, default=0.2,
                        help='read_latest_raw timeout in seconds. Default: 0.2.')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Display scale. Use 1.0 for lowest CPU. Default: 1.0.')
    parser.add_argument('--no-display', action='store_true',
                        help='Receive and drain latest frames without opening a window.')
    parser.add_argument('--opencv-threads', type=int, default=1,
                        help='OpenCV worker threads. Default: 1 to avoid oversubscription.')
    parser.add_argument('--strict-sync', action='store_true',
                        help='Require packet 1 to start every frame. Same as --sync-mode strict.')
    parser.add_argument('--sync-mode', choices=('wrap', 'strict', 'none'), default='wrap',
                        help='Frame split logic: wrap=packet number wraps, strict=packet 1 starts frame, none=accumulate rows by packet number.')
    parser.add_argument('--validate-frame-header', action='store_true',
                        help='Validate frame header and resolution in packet 1. Default: disabled.')
    parser.add_argument('--partial-refresh-ms', type=float, default=50.0,
                        help='Publish partial frames at this interval. Use 0 to disable. Default: 50 ms.')
    parser.add_argument('--min-partial-rows', type=int, default=1,
                        help='Minimum received rows before publishing a partial frame. Default: 1.')
    parser.add_argument('--keep-missing-rows', action='store_true',
                        help='Keep stale content in missing rows instead of clearing them to black.')
    parser.add_argument('--pixel-endian', choices=('big', 'little'), default='big',
                        help='Byte order of each RGB565 pixel in UDP payload. Default: big.')
    parser.add_argument('--color-order', choices=('rgb', 'bgr'), default='rgb',
                        help='Bit layout inside each 16-bit pixel: rgb=RGB565, bgr=BGR565. Default: rgb.')
    return parser.parse_args()


def maybe_resize(frame_bgr, scale):
    if scale == 1.0:
        return frame_bgr
    dst_w = max(1, int(frame_bgr.shape[1] * scale))
    dst_h = max(1, int(frame_bgr.shape[0] * scale))
    return cv2.resize(frame_bgr, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)


def print_stats(cam, display_frames, display_fps, rx_fps=None):
    stats = cam.stats()
    if rx_fps is None:
        rx_fps = stats['cap_fps']
    print(
        f"rx_frames={stats['frames_ok']} partial={stats['partial_frames']} "
        f"display_frames={display_frames} rx_fps={rx_fps:.1f} display_fps={display_fps:.1f} "
        f"packets={stats['packets_seen']} first={stats['first_packets']} "
        f"bad={stats['bad_packets']} incomplete={stats['incomplete_frames']} "
        f"last_rows={stats['last_partial_rows']} max_rows={stats['max_partial_rows']} "
        f"lost_rows={stats['partial_lost_rows']} overwritten={stats['frames_overwritten']} "
        f"last_pkt={stats['last_packet_num']} len={stats['last_packet_len']} "
        f"src={stats['last_src'] or '-'} last_error={stats['last_error'] or '-'}"
    )


def main():
    args = parse_args()

    cv2.setUseOptimized(True)
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)

    sync_mode = 'strict' if args.strict_sync else args.sync_mode

    cam = FPGAUDPCamera(
        port=args.port,
        local_ip=args.local_ip,
        fpga_ip=args.fpga_ip,
        filter_source=not args.no_filter,
        recv_buf_size=args.recv_buf_mb * 1024 * 1024,
        strict_frame_sync=(sync_mode == 'strict'),
        validate_frame_header=args.validate_frame_header,
        partial_publish_interval=args.partial_refresh_ms / 1000.0,
        min_partial_rows=args.min_partial_rows,
        clear_missing_rows=not args.keep_missing_rows,
        frame_boundary_mode=sync_mode,
        pixel_endian=args.pixel_endian,
        protocol=PROTOCOL,
    )

    print(f'Protocol: {WIDTH}x{HEIGHT}, {PROTOCOL.total_packets} packets/frame, UDP/{args.port}, RGB565')
    print('Receiver uses 3 raw RGB565 buffers. Display side always uses the latest complete or partial frame.')
    print(f"Frame sync mode: {sync_mode}")
    print(f"Frame header validation: {'enabled' if args.validate_frame_header else 'disabled'}")
    print(f"Partial frame publish: {args.partial_refresh_ms:g} ms, min_rows={args.min_partial_rows}, "
          f"missing_rows={'keep previous data' if args.keep_missing_rows else 'black'}")
    print(f"Pixel decode: endian={args.pixel_endian}, color_order={args.color_order}565 -> OpenCV BGR888")
    if not args.no_display:
        print('Press q or ESC in the image window to quit.')
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    display_frames = 0
    display_fps = 0.0
    last_display_time = time.time()
    last_stats_time = time.time()

    try:
        while True:
            result = cam.read_latest_raw(timeout=args.wait_timeout)
            now = time.time()

            if result is None:
                if now - last_stats_time >= args.stats_interval:
                    print_stats(cam, display_frames, display_fps)
                    last_stats_time = now
                continue

            raw565, _, rx_fps = result

            if not args.no_display:
                frame_bgr = FPGAUDPCamera.rgb565_to_bgr888(raw565, color_order=args.color_order)
                frame_bgr = maybe_resize(frame_bgr, args.scale)
                cv2.imshow(WINDOW_NAME, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

            display_frames += 1
            display_fps = 1.0 / max(now - last_display_time, 1e-6)
            last_display_time = now

            if now - last_stats_time >= args.stats_interval:
                print_stats(cam, display_frames, display_fps, rx_fps)
                last_stats_time = now

    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        cam.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

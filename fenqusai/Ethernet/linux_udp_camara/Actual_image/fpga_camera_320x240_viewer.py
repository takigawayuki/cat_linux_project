#!/usr/bin/env python3
"""
FPGA UDP camera viewer for the 320x240 actual-image protocol.

Protocol:
  - 320x240 RGB565 image.
  - 270 UDP packets per frame, first 240 image rows are displayed.
  - Packet 1 payload: pkt_num(4) + frame_header(4) + resolution(4) + row0(640) = 652 bytes.
  - Packet 2..270 payload: pkt_num(4) + row(640) = 644 bytes.
  - FPGA sends big-endian bytes, Linux decodes RGB565 -> OpenCV BGR888.
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


PROTOCOL = FPGAUDPImageProtocol(width=320, height=240, packets_per_frame=270)
WIDTH = PROTOCOL.width
HEIGHT = PROTOCOL.height
PACKETS_PER_FRAME = PROTOCOL.total_packets
ROW_BYTES = PROTOCOL.row_bytes
FIRST_PACKET_BYTES = PROTOCOL.first_packet_bytes
ROW_PACKET_BYTES = PROTOCOL.row_packet_bytes
WINDOW_NAME = 'FPGA UDP Camera 320x240'


def parse_args():
    parser = argparse.ArgumentParser(
        description='FPGA UDP actual-image viewer, 320x240 RGB565.'
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
    parser.add_argument('--partial-refresh-ms', type=float, default=0.0,
                        help='Publish partial frames at this interval. Use 0 to display only complete frames. Default: 0 ms.')
    parser.add_argument('--min-partial-rows', type=int, default=1,
                        help='Minimum received rows before publishing a partial frame. Default: 1.')
    parser.add_argument('--keep-missing-rows', action='store_true',
                        help='Keep stale content in missing rows instead of clearing them to black.')
    parser.add_argument('--packet-number-mode', choices=('auto', 'header', 'footer', 'sequential'), default='auto',
                        help='Packet number parsing. Default auto tries header, footer, then sequential fallback.')
    parser.add_argument('--payload-offset', type=int, default=None,
                        help='Override RGB565 payload offset. Default: auto.')
    return parser.parse_args()


def maybe_resize(frame_bgr, scale):
    if scale == 1.0:
        return frame_bgr
    dst_w = max(1, int(frame_bgr.shape[1] * scale))
    dst_h = max(1, int(frame_bgr.shape[0] * scale))
    return cv2.resize(frame_bgr, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)


def pixel_stats(raw565):
    if raw565 is None:
        return 'raw=-'
    nonzero = int((raw565 != 0).sum())
    return (
        f"raw_nz={nonzero}/{raw565.size} raw_min=0x{int(raw565.min()):04x} "
        f"raw_max=0x{int(raw565.max()):04x} raw_mean={float(raw565.mean()):.1f}"
    )


def print_stats(cam, display_frames, display_fps, rx_fps=None, raw565=None):
    stats = cam.stats()
    if rx_fps is None:
        rx_fps = stats['cap_fps']
    print(
        f"rx_frames={stats['frames_ok']} partial={stats['partial_frames']} "
        f"display_frames={display_frames} rx_fps={rx_fps:.1f} display_fps={display_fps:.1f} "
        f"packets={stats['packets_seen']} first={stats['first_packets']} "
        f"bad={stats['bad_packets']} extra={stats['extra_packets']} fallback={stats['packet_number_fallbacks']} incomplete={stats['incomplete_frames']} "
        f"last_rows={stats['last_partial_rows']} max_rows={stats['max_partial_rows']} "
        f"lost_rows={stats['partial_lost_rows']} overwritten={stats['frames_overwritten']} "
        f"last_pkt={stats['last_packet_num']} pkt_src={stats['last_packet_number_source'] or '-'} len={stats['last_packet_len']} "
        f"src={stats['last_src'] or '-'} {pixel_stats(raw565)} "
        f"last_error={stats['last_error'] or '-'}"
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
        pixel_endian='big',
        packet_number_mode=args.packet_number_mode,
        payload_offset=args.payload_offset,
        protocol=PROTOCOL,
    )

    print(f'Protocol: {WIDTH}x{HEIGHT}, {PACKETS_PER_FRAME} packets/frame, UDP/{args.port}, RGB565 big-endian')
    print(f'Packet bytes: first={FIRST_PACKET_BYTES}, row={ROW_PACKET_BYTES}, row_payload={ROW_BYTES}')
    print(f'Expected stream: 60 fps, about {PROTOCOL.expected_packets_per_second} UDP packets/s')
    if PROTOCOL.total_packets != HEIGHT:
        print(f'Packet note: display height is {HEIGHT}, packets_per_frame is {PROTOCOL.total_packets}; packets above {HEIGHT} are counted as extra and not drawn')
    if args.partial_refresh_ms > 0:
        print('Receiver uses 3 raw RGB565 buffers. Display side uses the latest complete or partial frame.')
    else:
        print('Receiver uses 3 raw RGB565 buffers. Display side uses complete frames only.')
    print(f'Frame sync mode: {sync_mode}')
    print(f"Packet number mode: {args.packet_number_mode}, payload offset={args.payload_offset if args.payload_offset is not None else 'auto'}")
    print(f"Frame header validation: {'enabled' if args.validate_frame_header else 'disabled'}")
    print(f"Partial frame publish: {args.partial_refresh_ms:g} ms, min_rows={args.min_partial_rows}, "
          f"missing_rows={'keep previous data' if args.keep_missing_rows else 'black'}")
    print('Pixel decode: endian=big, color_order=rgb565 -> OpenCV BGR888')
    if not args.no_display:
        print('Press q or ESC in the image window to quit.')
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    display_frames = 0
    display_fps = 0.0
    last_display_time = time.time()
    last_stats_time = time.time()
    last_raw565 = None

    try:
        while True:
            result = cam.read_latest_raw(timeout=args.wait_timeout)
            now = time.time()

            if result is None:
                if now - last_stats_time >= args.stats_interval:
                    print_stats(cam, display_frames, display_fps, raw565=last_raw565)
                    last_stats_time = now
                continue

            raw565, _, rx_fps = result
            last_raw565 = raw565

            if not args.no_display:
                frame_bgr = FPGAUDPCamera.rgb565_to_bgr888(raw565, color_order='rgb')
                frame_bgr = maybe_resize(frame_bgr, args.scale)
                cv2.imshow(WINDOW_NAME, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

            display_frames += 1
            display_fps = 1.0 / max(now - last_display_time, 1e-6)
            last_display_time = now

            if now - last_stats_time >= args.stats_interval:
                print_stats(cam, display_frames, display_fps, rx_fps, raw565=last_raw565)
                last_stats_time = now

    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        cam.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

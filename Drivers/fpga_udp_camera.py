"""
Generic UDP receiver for FPGA row-based RGB565 image streams.

This module is intentionally protocol-parameter driven. Project-specific values
such as image width, height, and packet count should be provided by the caller
through FPGAUDPImageProtocol or width/height arguments.

The receiver thread only receives UDP packets and assembles raw RGB565 rows.
RGB565 -> BGR888 conversion is done by read()/read_latest() in the caller side.
"""
from dataclasses import dataclass
import socket
import struct
import threading
import time

import numpy as np


PORT = 1234
FPGA_IP = '192.168.1.11'
FRAME_HEADER = 0xF05AA50F
BUFFER_COUNT = 3
RECV_PACKET_BYTES = 4096
SO_RCVBUFFORCE = 33


@dataclass(frozen=True)
class FPGAUDPImageProtocol:
    width: int
    height: int
    frame_header: int = FRAME_HEADER
    packet_num_bytes: int = 4
    frame_header_bytes: int = 4
    resolution_bytes: int = 4
    pixel_bytes: int = 2

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError('width and height must be positive')
        if self.pixel_bytes != 2:
            raise ValueError('only RGB565/BGR565 2-byte pixels are supported')
        if self.packet_num_bytes != 4:
            raise ValueError('only 4-byte packet numbers are supported')
        if self.frame_header_bytes != 4:
            raise ValueError('only 4-byte frame headers are supported')
        if self.resolution_bytes != 4:
            raise ValueError('only 4-byte resolution fields are supported')

    @property
    def total_packets(self):
        return self.height

    @property
    def row_bytes(self):
        return self.width * self.pixel_bytes

    @property
    def first_packet_bytes(self):
        return (self.packet_num_bytes + self.frame_header_bytes +
                self.resolution_bytes + self.row_bytes)

    @property
    def row_packet_bytes(self):
        return self.packet_num_bytes + self.row_bytes

    @property
    def expected_packets_per_second(self):
        return self.total_packets * 60


class FPGAUDPCamera:
    def __init__(self, port=PORT, local_ip='', fpga_ip=FPGA_IP,
                 filter_source=True, recv_buf_size=64 * 1024 * 1024,
                 strict_frame_sync=False, validate_frame_header=False,
                 partial_publish_interval=0.0, min_partial_rows=1,
                 clear_missing_rows=True, frame_boundary_mode=None,
                 publish_partial_on_boundary=True, pixel_endian='big',
                 protocol=None, width=None, height=None,
                 frame_header=FRAME_HEADER):
        if protocol is None:
            if width is None or height is None:
                raise ValueError('FPGAUDPCamera requires protocol or width/height')
            protocol = FPGAUDPImageProtocol(
                width=int(width),
                height=int(height),
                frame_header=int(frame_header),
            )
        self.protocol = protocol
        self._width = protocol.width
        self._height = protocol.height
        self._total_packets = protocol.total_packets
        self._row_bytes = protocol.row_bytes
        self._first_packet_bytes = protocol.first_packet_bytes
        self._row_packet_bytes = protocol.row_packet_bytes
        self._frame_header = protocol.frame_header

        self._raw_buffers = [
            np.zeros((self._height, self._width), dtype=np.uint16)
            for _ in range(BUFFER_COUNT)
        ]
        self._write_idx = 0
        self._latest_idx = None
        self._latest_seq = 0
        self._last_read_seq = 0
        self._roi = (0, 0, self._width, self._height)

        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = threading.Event()

        self.cap_fps = 0.0
        self.frames_ok = 0
        self.partial_frames = 0
        self.frames_overwritten = 0
        self.packets_seen = 0
        self.first_packets = 0
        self.ignored_packets = 0
        self.bad_packets = 0
        self.incomplete_frames = 0
        self.partial_lost_rows = 0
        self.max_partial_rows = 0
        self.last_partial_rows = 0
        self.last_packet_num = 0
        self.last_packet_len = 0
        self.last_src = ''
        self.last_error = ''

        self._fpga_ip = fpga_ip
        self._filter_source = filter_source
        if frame_boundary_mode is None:
            frame_boundary_mode = 'strict' if strict_frame_sync else 'wrap'
        if frame_boundary_mode not in ('strict', 'wrap', 'none'):
            raise ValueError('frame_boundary_mode must be strict, wrap, or none')

        self._strict_frame_sync = (frame_boundary_mode == 'strict')
        self._validate_frame_header = validate_frame_header
        self._partial_publish_interval = float(partial_publish_interval)
        self._min_partial_rows = int(min_partial_rows)
        self._clear_missing_rows = clear_missing_rows
        self._frame_boundary_mode = frame_boundary_mode
        self._publish_partial_on_boundary = publish_partial_on_boundary
        if pixel_endian not in ('big', 'little'):
            raise ValueError('pixel_endian must be big or little')
        self._pixel_endian = pixel_endian
        self._pixel_dtype = np.dtype('>u2' if pixel_endian == 'big' else '<u2')

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, SO_RCVBUFFORCE, recv_buf_size)
        except OSError:
            try:
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf_size)
            except OSError:
                pass
        self._sock.settimeout(0.5)
        self._sock.bind((local_ip, port))

        actual = self._sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        bind_ip = local_ip or '0.0.0.0'
        print(f'FPGAUDPCamera listening on {bind_ip}:{port}, '
              f'SO_RCVBUF={actual // 1024} KB')

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    @staticmethod
    def rgb565_to_bgr888(raw565, color_order='rgb'):
        if color_order not in ('rgb', 'bgr'):
            raise ValueError('color_order must be rgb or bgr')

        high5 = ((raw565 >> 11) & 0x1F).astype(np.uint8)
        g6 = ((raw565 >> 5) & 0x3F).astype(np.uint8)
        low5 = (raw565 & 0x1F).astype(np.uint8)

        if color_order == 'rgb':
            r5 = high5
            b5 = low5
        else:
            b5 = high5
            r5 = low5

        b = (b5 << 3) | (b5 >> 2)
        g = (g6 << 2) | (g6 >> 4)
        r = (r5 << 3) | (r5 >> 2)
        return np.dstack((b, g, r)).astype(np.uint8)

    def _next_write_idx_locked(self):
        next_idx = (self._write_idx + 1) % BUFFER_COUNT
        if self._latest_idx is not None and next_idx == self._latest_idx:
            next_idx = (next_idx + 1) % BUFFER_COUNT
        self._write_idx = next_idx
        return next_idx

    def _next_write_idx(self):
        with self._lock:
            return self._next_write_idx_locked()

    def _mark_published_locked(self, now, last_time, partial):
        if self._latest_idx is not None and self._latest_seq != self._last_read_seq:
            self.frames_overwritten += 1
        self._latest_seq += 1
        self.cap_fps = 1.0 / max(now - last_time, 1e-6)
        if partial:
            self.partial_frames += 1
        else:
            self.frames_ok += 1

    def _publish_complete_frame(self, raw_idx, last_time):
        now = time.time()
        with self._lock:
            self._latest_idx = raw_idx
            self._mark_published_locked(now, last_time, partial=False)
        self._event.set()
        return now

    def _publish_partial_snapshot(self, raw, received, raw_idx, last_time):
        now = time.time()
        with self._lock:
            snap_idx = (raw_idx + 1) % BUFFER_COUNT
            if snap_idx == raw_idx:
                snap_idx = (snap_idx + 1) % BUFFER_COUNT
            if self._latest_idx is not None and snap_idx == self._latest_idx:
                snap_idx = (snap_idx + 1) % BUFFER_COUNT
                if snap_idx == raw_idx:
                    snap_idx = (snap_idx + 1) % BUFFER_COUNT

            dst = self._raw_buffers[snap_idx]
            dst[:, :] = raw
            if self._clear_missing_rows:
                dst[~received, :] = 0
            self._latest_idx = snap_idx
            self._mark_published_locked(now, last_time, partial=True)
        self._event.set()
        return now

    def _mark_bad_packet(self, reason):
        self.bad_packets += 1
        self.last_error = reason

    def _mark_incomplete_frame(self, rows_received):
        self.incomplete_frames += 1
        self.last_partial_rows = rows_received
        if rows_received > self.max_partial_rows:
            self.max_partial_rows = rows_received
        self.partial_lost_rows += self._height - rows_received

    def _recv_loop(self):
        received = np.zeros(self._height, dtype=bool)
        rx_buf = bytearray(max(RECV_PACKET_BYTES, self._first_packet_bytes + 64))
        rows_received = 0
        raw_idx = self._write_idx
        raw = self._raw_buffers[raw_idx]
        raw[:, :] = 0
        last_time = time.time()
        last_partial_publish = last_time
        prev_pkt = 0
        complete_published = False

        unpack_u32 = struct.Struct('!I').unpack_from
        unpack_header = struct.Struct('!IIHH').unpack_from

        while not self._stop.is_set():
            try:
                nbytes, addr = self._sock.recvfrom_into(rx_buf)
            except socket.timeout:
                continue
            except OSError:
                break

            src_ip, _ = addr
            if self._filter_source and self._fpga_ip and src_ip != self._fpga_ip:
                self.ignored_packets += 1
                continue

            self.packets_seen += 1
            self.last_packet_len = nbytes
            self.last_src = src_ip

            if nbytes < 4:
                self._mark_bad_packet(f'short packet: len={nbytes}')
                continue

            pkt = unpack_u32(rx_buf, 0)[0]
            self.last_packet_num = pkt
            if pkt < 1 or pkt > self._total_packets:
                self._mark_bad_packet(f'bad packet number: pkt={pkt}, len={nbytes}')
                continue

            started_new_frame = False
            if (self._frame_boundary_mode == 'wrap' and rows_received > 0 and
                    prev_pkt and pkt <= prev_pkt):
                if (self._publish_partial_on_boundary and
                        self._partial_publish_interval > 0 and
                        rows_received >= self._min_partial_rows):
                    last_time = self._publish_partial_snapshot(raw, received, raw_idx, last_time)
                    last_partial_publish = last_time
                self._mark_incomplete_frame(rows_received)
                raw_idx = self._next_write_idx()
                raw = self._raw_buffers[raw_idx]
                raw[:, :] = 0
                received[:] = False
                rows_received = 0
                complete_published = False
                last_partial_publish = time.time()
                started_new_frame = True

            if rows_received == 0 and not started_new_frame:
                raw_idx = self._next_write_idx()
                raw = self._raw_buffers[raw_idx]
                raw[:, :] = 0

            if pkt == 1:
                self.first_packets += 1

                if nbytes >= self._first_packet_bytes:
                    if self._validate_frame_header:
                        _, frame_header, width, height = unpack_header(rx_buf, 0)
                        if (frame_header != self._frame_header or
                                width != self._width or height != self._height):
                            self._mark_bad_packet(
                                f'header mismatch: frame_header=0x{frame_header:08x}, '
                                f'width={width}, height={height}'
                            )
                            continue
                    pixel_offset = 12
                elif nbytes >= self._row_packet_bytes:
                    pixel_offset = 4
                else:
                    self._mark_bad_packet(
                        f'short packet 1: len={nbytes}, '
                        f'expected={self._row_packet_bytes} or {self._first_packet_bytes}'
                    )
                    continue

                if self._strict_frame_sync and rows_received:
                    self._mark_incomplete_frame(rows_received)
                    raw_idx = self._next_write_idx()
                    raw = self._raw_buffers[raw_idx]
                    raw[:, :] = 0
                    received[:] = False
                    rows_received = 0
                    complete_published = False

                raw[0, :] = np.frombuffer(
                    rx_buf, dtype=self._pixel_dtype, count=self._width,
                    offset=pixel_offset,
                )
                if not received[0]:
                    received[0] = True
                    rows_received += 1

            else:
                if self._strict_frame_sync and rows_received == 0:
                    continue
                if nbytes < self._row_packet_bytes:
                    self._mark_bad_packet(
                        f'short row packet: pkt={pkt}, len={nbytes}, '
                        f'expected={self._row_packet_bytes}'
                    )
                    continue

                row = pkt - 1
                raw[row, :] = np.frombuffer(
                    rx_buf, dtype=self._pixel_dtype, count=self._width, offset=4,
                )
                if not received[row]:
                    received[row] = True
                    rows_received += 1

            prev_pkt = pkt
            self.last_partial_rows = rows_received
            if rows_received > self.max_partial_rows:
                self.max_partial_rows = rows_received

            if rows_received == self._height and not complete_published:
                last_time = self._publish_complete_frame(raw_idx, last_time)
                last_partial_publish = last_time
                complete_published = True
                if self._frame_boundary_mode != 'none':
                    received[:] = False
                    rows_received = 0
                    complete_published = False
            elif (self._partial_publish_interval > 0 and
                  rows_received >= self._min_partial_rows):
                now = time.time()
                if now - last_partial_publish >= self._partial_publish_interval:
                    last_time = self._publish_partial_snapshot(raw, received, raw_idx, last_time)
                    last_partial_publish = last_time

    def _copy_latest_raw(self, wait_timeout=None):
        got_frame = self._event.wait(wait_timeout)
        self._event.clear()

        with self._lock:
            if self._latest_idx is None:
                return None, False
            raw = self._raw_buffers[self._latest_idx].copy()
            self._last_read_seq = self._latest_seq
            roi = self._roi
            fps = self.cap_fps

        return (raw, roi, fps), got_frame

    def read_raw(self):
        """Block until next complete or partial raw frame."""
        while True:
            result, _ = self._copy_latest_raw(None)
            if result is not None:
                return result

    def read_latest_raw(self, timeout=0.5):
        """Wait up to timeout for a new raw frame, then return latest raw frame."""
        result, got_frame = self._copy_latest_raw(timeout)
        if result is None:
            return None
        if not got_frame and self.frames_ok == 0 and self.partial_frames == 0:
            return None
        return result

    def read(self):
        """Block until next frame. Returns (bgr_frame, roi, cap_fps)."""
        raw, roi, fps = self.read_raw()
        return self.rgb565_to_bgr888(raw), roi, fps

    def read_latest(self, timeout=0.5):
        """Wait up to timeout for a new frame, then return latest BGR frame."""
        result = self.read_latest_raw(timeout)
        if result is None:
            return None
        raw, roi, fps = result
        return self.rgb565_to_bgr888(raw), roi, fps

    def stats(self):
        with self._lock:
            return {
                'frames_ok': self.frames_ok,
                'partial_frames': self.partial_frames,
                'frames_overwritten': self.frames_overwritten,
                'cap_fps': self.cap_fps,
                'packets_seen': self.packets_seen,
                'first_packets': self.first_packets,
                'ignored_packets': self.ignored_packets,
                'bad_packets': self.bad_packets,
                'incomplete_frames': self.incomplete_frames,
                'partial_lost_rows': self.partial_lost_rows,
                'max_partial_rows': self.max_partial_rows,
                'last_partial_rows': self.last_partial_rows,
                'last_packet_num': self.last_packet_num,
                'last_packet_len': self.last_packet_len,
                'last_src': self.last_src,
                'last_error': self.last_error,
                'frame_boundary_mode': self._frame_boundary_mode,
                'pixel_endian': self._pixel_endian,
                'width': self._width,
                'height': self._height,
                'total_packets': self._total_packets,
                'row_bytes': self._row_bytes,
                'row_packet_bytes': self._row_packet_bytes,
                'first_packet_bytes': self._first_packet_bytes,
                'expected_packets_per_second': self.protocol.expected_packets_per_second,
            }

    def release(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

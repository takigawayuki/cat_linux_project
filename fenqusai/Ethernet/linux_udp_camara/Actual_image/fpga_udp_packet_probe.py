#!/usr/bin/env python3
"""
Probe FPGA UDP packet numbers and payload lengths without assembling frames.

Use this when tcpdump shows packets but the receiver cannot build a full frame.
It prints the real packet-number distribution so we can verify whether FPGA sends
1..270, only part of the rows, or another numbering scheme.
"""

import argparse
import socket
import time
from collections import Counter


PORT = 1234
FPGA_IP = "192.168.1.11"
TOTAL_PACKETS = 270
FRAME_HEADER = 0xF05AA50F


def parse_args():
    parser = argparse.ArgumentParser(description="Probe FPGA UDP packet numbers.")
    parser.add_argument("--local-ip", default="", help="Local IP to bind. Default: all interfaces.")
    parser.add_argument("--port", type=int, default=PORT, help="UDP port. Default: 1234.")
    parser.add_argument("--fpga-ip", default=FPGA_IP, help="FPGA source IP. Default: 192.168.1.11.")
    parser.add_argument("--no-filter", action="store_true", help="Do not filter by source IP.")
    parser.add_argument("--seconds", type=float, default=3.0, help="Probe duration. Default: 3 seconds.")
    parser.add_argument("--samples", type=int, default=12, help="Number of packet samples to print.")
    return parser.parse_args()


def main():
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
    sock.bind((args.local_ip, args.port))
    sock.settimeout(0.5)

    filter_ip = "" if args.no_filter else args.fpga_ip
    deadline = time.time() + args.seconds

    packet_nums = Counter()
    lengths = Counter()
    first_packets = 0
    header_ok = 0
    bad_packet_nums = 0
    samples = []
    total = 0
    ignored = 0

    print(f"Listening on {args.local_ip or '0.0.0.0'}:{args.port} for {args.seconds:.1f}s")
    if filter_ip:
        print(f"Filtering source IP: {filter_ip}")
    else:
        print("Source-IP filtering disabled")

    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue

        src_ip, src_port = addr
        if filter_ip and src_ip != filter_ip:
            ignored += 1
            continue

        total += 1
        lengths[len(data)] += 1
        if len(data) < 4:
            bad_packet_nums += 1
            continue

        pkt = int.from_bytes(data[0:4], "big")
        packet_nums[pkt] += 1

        if len(samples) < args.samples:
            samples.append((pkt, len(data), src_ip, src_port, data[:24].hex(" ")))

        if pkt == 1:
            first_packets += 1
            if len(data) >= 12:
                frame_header = int.from_bytes(data[4:8], "big")
                width = int.from_bytes(data[8:10], "big")
                height = int.from_bytes(data[10:12], "big")
                if frame_header == FRAME_HEADER and width == 480 and height == 270:
                    header_ok += 1

    sock.close()

    in_range = sorted(k for k in packet_nums if 1 <= k <= TOTAL_PACKETS)
    missing = [k for k in range(1, TOTAL_PACKETS + 1) if k not in packet_nums]
    even = sum(v for k, v in packet_nums.items() if 1 <= k <= TOTAL_PACKETS and k % 2 == 0)
    odd = sum(v for k, v in packet_nums.items() if 1 <= k <= TOTAL_PACKETS and k % 2 == 1)

    print("\nSummary")
    print(f"total_packets={total} ignored={ignored} bad_packet_nums={bad_packet_nums}")
    print(f"length_counts={dict(sorted(lengths.items()))}")
    print(f"unique_packet_nums={len(packet_nums)} in_range_unique={len(in_range)}")
    print(f"first_packets={first_packets} header_ok={header_ok}")
    print(f"odd_packets={odd} even_packets={even}")
    if in_range:
        print(f"min_pkt={in_range[0]} max_pkt={in_range[-1]}")
    print(f"missing_count={len(missing)}")
    print(f"missing_first_40={missing[:40]}")

    print("\nMost common packet numbers")
    for pkt, count in packet_nums.most_common(20):
        print(f"pkt={pkt} count={count}")

    print("\nSamples")
    for pkt, length, src_ip, src_port, head in samples:
        print(f"pkt={pkt} len={length} src={src_ip}:{src_port} head={head}")


if __name__ == "__main__":
    main()

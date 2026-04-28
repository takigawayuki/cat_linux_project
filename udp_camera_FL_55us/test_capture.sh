#!/bin/bash
# 抓取一个包并显示前 20 字节的十六进制内容
sudo timeout 5 tcpdump -i eth1 -c 1 -X udp port 1234 2>/dev/null | head -30

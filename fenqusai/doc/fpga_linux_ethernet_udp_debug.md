# FPGA 与 Linux 以太网 UDP 联调记录

本文档用于 FPGA 通过以太网向 Linux 主机发送 UDP 图像数据时的现场联调、抓包和排错。

当前工程里常用参数：

| 项目 | 当前常见值 |
| --- | --- |
| UDP 端口 | `1234` |
| Linux/RK 接收 IP | `192.168.1.22` |
| FPGA IP | `192.168.1.11` |
| 图像格式 | RGB565，大端序 |
| 包头 | UDP payload 前 4 字节为包序号，大端序 |
| 帧头 | `0xF05AA50F`，常出现在第 1 包 payload 内 |
| 常见分辨率 | `640x480`、`720x480`、`320x240` |

## 1. 接线与网卡确认

先确认 Linux 能看到网卡，并且网线连接状态正常。

```bash
ip link
ip addr
```

查看指定网卡状态，例如网卡名为 `eth1`：

```bash
ip link show eth1
ethtool eth1
```

重点看：

- `state UP`：网卡已启用。
- `Link detected: yes`：物理链路已连接。
- `Speed` / `Duplex`：速率和双工模式正常。

如果网卡没有启用：

```bash
sudo ip link set eth1 up
```

## 2. Linux 静态 IP 配置

FPGA 直连 Linux 网口时，建议双方配置到同一个网段。

示例：

| 设备 | IP |
| --- | --- |
| RK/Linux | `192.168.1.22` |
| FPGA | `192.168.1.11` |

临时配置 Linux 网卡：

```bash
sudo ip addr flush dev eth1
sudo ip addr add 192.168.1.22/24 dev eth1
sudo ip link set eth1 up
```

确认配置：

```bash
ip addr show eth1
ip route
```

如果 FPGA 支持 ICMP，可以测试：

```bash
ping 192.168.1.11
```

注意：很多 FPGA UDP 发送逻辑不实现 ping，所以 ping 不通不一定代表 UDP 不通。UDP 联调以 `tcpdump` 抓包结果为准。

当前环境检测结果：

- `eth1` 已配置 `192.168.1.22/24`，网卡状态为 UP。
- 到 FPGA `192.168.1.11` 的路由为 `dev eth1 src 192.168.1.22`。
- `ping 192.168.1.11` 当前 2 包无响应；如果 FPGA 未实现 ICMP，这是正常现象，继续用 `tcpdump` 判断 UDP 是否到达。

## 3. 运行本工程 UDP 接收程序

### 3.1 640x480 接收程序

对应目录：

```bash
cd udp_camera_FL
make
./udp_receiver
```

程序默认监听：

```text
0.0.0.0:1234
```

程序启动后应该能看到类似：

```text
UDP接收启动: port=1234
```

收到完整帧后，会打印帧计数、接收行数和 FPS。

### 3.2 720x480 接收程序

C++ 版本：

```bash
cd udp_image_HL_720x480/CPP
make
./udp_receiver
```

Python 版本：

```bash
cd udp_image_HL_720x480
python3 udp_image_receiver.py
```

Python 版本会打印：

```text
监听 0.0.0.0:1234
收到UDP包: ... bytes 来自 ...
```

如果只想确认是否有 UDP 包到达，Python 版本的日志比较直观；如果追求实时显示性能，优先用 C++ 版本。

## 4. 确认 Linux 是否监听 UDP 端口

运行接收程序后，在另一个终端查看端口：

```bash
ss -lunp | grep ':1234'
```

或者：

```bash
sudo netstat -lunp | grep ':1234'
```

能看到 `0.0.0.0:1234` 或本机 IP 的 `:1234`，说明接收程序已经绑定端口。

## 5. 抓包确认 FPGA UDP 是否到达 Linux

### 5.1 最常用命令

抓指定网卡、指定 UDP 端口：

```bash
sudo tcpdump -i eth1 -nn udp port 1234
```

如果不知道 FPGA 发到哪个端口，可以先抓所有 UDP：

```bash
sudo tcpdump -i eth1 -nn udp
```

如果不知道是哪块网卡，先用 `any`：

```bash
sudo tcpdump -i any -nn udp port 1234
```

看到下面这种输出是正常的，表示 `tcpdump` 已经开始监听网卡，不是报错：

```text
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on eth1, link-type EN10MB (Ethernet), capture size 262144 bytes
```

含义：

- `verbose output suppressed`：当前没有开详细解析模式，需要更详细协议内容时加 `-v` 或 `-vv`。
- `listening on eth1`：正在监听 `eth1` 网卡。
- 如果后面一直没有新的数据行，说明当前没有抓到符合过滤条件的包。

例如执行下面命令后只停在 `listening on eth1...`：

```bash
sudo tcpdump -i eth1 host 192.168.1.11
```

通常表示还没有抓到与 FPGA `192.168.1.11` 相关的数据。此时建议换成更明确的 UDP 抓包命令：

```bash
sudo tcpdump -i eth1 -nn -vv -X src host 192.168.1.11 and udp
```

或者直接抓 UDP 端口：

```bash
sudo tcpdump -i eth1 -nn -vv -X udp port 1234
```

### 5.2 抓包并显示十六进制内容

查看 UDP payload，确认包序号、帧头是否正确：

```bash
sudo tcpdump -i eth1 -nn -vv -X udp port 1234 -c 10
```

如果只抓 1 个包并显示前面内容：

```bash
sudo timeout 5 tcpdump -i eth1 -c 1 -X udp port 1234
```

正常情况下，payload 前 4 字节是包序号：

```text
00 00 00 01
```

第 1 包后面通常应能看到帧头：

```text
f0 5a a5 0f
```

如果看到包序号是反的，例如 `01 00 00 00`，说明 FPGA 端字节序可能不对。

### 5.3 抓源 IP / 目的 IP / 端口

只抓 FPGA 源 IP：

```bash
sudo tcpdump -i eth1 -nn src host 192.168.1.11
```

只抓发往 Linux 的 UDP 端口：

```bash
sudo tcpdump -i eth1 -nn dst host 192.168.1.22 and udp dst port 1234
```

同时显示二层 MAC 地址：

```bash
sudo tcpdump -i eth1 -nn -e udp port 1234
```

如果能看到 FPGA 的源 MAC，说明至少二层以太网已经到 Linux 网卡。

### 5.4 保存 pcap 文件给 Wireshark 分析

```bash
sudo tcpdump -i eth1 -nn udp port 1234 -w fpga_udp_1234.pcap
```

抓 10 秒自动结束：

```bash
sudo timeout 10 tcpdump -i eth1 -nn udp port 1234 -w fpga_udp_1234.pcap
```

用 Wireshark 打开：

```bash
wireshark fpga_udp_1234.pcap
```

没有图形界面时可以用 `tshark`：

```bash
tshark -r fpga_udp_1234.pcap -Y "udp.port == 1234"
```

## 6. 判断问题位置

### 6.1 tcpdump 抓不到任何包

优先检查：

- 网卡名是否选对：`ip addr` 查看真实网卡名。
- 网线、交换机、FPGA PHY 是否正常。
- Linux 网卡是否 UP：`ip link show eth1`。
- FPGA 目的 MAC 是否正确。
- FPGA 目的 IP 是否为 Linux 当前 IP。
- FPGA 目的 UDP 端口是否为 `1234`。
- FPGA 是否真的开始发送。

继续看网卡统计：

```bash
ip -s link show eth1
ethtool -S eth1
```

如果 `RX packets` 增加，但 tcpdump 无显示，检查抓包网卡或过滤条件。

### 6.2 tcpdump 能抓到，但程序收不到

优先检查：

- 程序是否已经启动并监听 `1234`：

```bash
ss -lunp | grep ':1234'
```

- FPGA 发的目的端口是否真的是 `1234`。
- UDP 校验和是否异常。
- Linux 防火墙是否拦截。

临时查看防火墙：

```bash
sudo iptables -L -n -v
sudo nft list ruleset
```

如果只是实验环境，可以临时放行 UDP 1234：

```bash
sudo iptables -I INPUT -p udp --dport 1234 -j ACCEPT
```

### 6.3 程序能收到包，但画面异常

优先检查 FPGA 与 Linux 接收程序的协议是否一致：

- 分辨率是否一致：`640x480`、`720x480`、`320x240` 不要混用。
- 每包像素数量是否一致。
- RGB565 是否为大端序。
- 第 1 包是否带 `0xF05AA50F + width + height`。
- 包序号是否从 `1` 开始。
- 最后一包是否为 ROI 坐标包。
- FPGA 是否有丢包、乱序或重复包。

## 7. 当前工程包格式参考

### 7.1 640x480 行包格式

对应代码：`udp_camera_FL/udp_fpga_receiver.cpp`

端口：

```text
UDP 1234
```

图像：

```text
WIDTH  = 640
HEIGHT = 480
TOTAL_PACKETS = 481
```

包格式：

```text
第 1 包:
  packet_num: 4 字节，大端，值为 1
  frame_head: 4 字节，大端，通常为 0xF05AA50F
  width:      2 字节，大端
  height:     2 字节，大端
  row0:       640 个 RGB565 像素，每像素 2 字节，大端

第 2-480 包:
  packet_num: 4 字节，大端，值为 2-480
  row:        640 个 RGB565 像素，每像素 2 字节，大端

第 481 包:
  packet_num: 4 字节，大端，值为 481
  x1:         2 字节，大端
  y1:         2 字节，大端
  x2:         2 字节，大端
  y2:         2 字节，大端
```

### 7.2 720x480 半行包格式

对应代码：`udp_image_HL_720x480/udp_image_receiver.py`、`udp_image_HL_720x480/CPP/udp_fpga_receiver.cpp`

图像：

```text
WIDTH  = 720
HEIGHT = 480
```

每行拆成左右两包：

```text
row = (packet_num - 1) / 2
col_start = ((packet_num - 1) % 2) * (WIDTH / 2)
```

也就是：

```text
packet 1 -> 第 0 行左半部分
packet 2 -> 第 0 行右半部分
packet 3 -> 第 1 行左半部分
packet 4 -> 第 1 行右半部分
...
```

最后 ROI 包 payload 通常为 8 或 16 字节，前 8 字节为：

```text
x1 y1 x2 y2
```

每个坐标为 2 字节大端。

## 8. Linux 接收性能参数

UDP 高帧率传图时，Linux 默认接收缓冲可能偏小，可以临时调大：

```bash
sudo sysctl -w net.core.rmem_max=67108864
sudo sysctl -w net.core.rmem_default=67108864
```

查看当前值：

```bash
sysctl net.core.rmem_max
sysctl net.core.rmem_default
```

接收程序里也会设置 socket 接收缓冲，例如：

```text
SO_RCVBUF / SO_RCVBUFFORCE
```

如果仍然丢包，可以：

- 降低 FPGA 发送速率做对照。
- 增大 UDP 包间隔。
- 用 C++ 接收程序替代 Python 接收程序。
- 减少接收端打印日志，频繁打印会显著影响性能。
- 尽量直连网口或使用稳定交换机。

## 9. 快速联调流程

现场建议按这个顺序走：

```bash
# 1. 看网卡
ip addr
ip link show eth1
ethtool eth1

# 2. 配 Linux IP
sudo ip addr flush dev eth1
sudo ip addr add 192.168.1.22/24 dev eth1
sudo ip link set eth1 up

# 3. 先抓包，不开接收程序也可以抓
sudo tcpdump -i eth1 -nn -vv -X udp port 1234 -c 5

# 4. 如果能抓到包，再启动工程接收程序
cd udp_camera_FL
make
./udp_receiver

# 5. 另开终端确认端口监听
ss -lunp | grep ':1234'
```

判断结果：

| 现象 | 结论 |
| --- | --- |
| `tcpdump` 抓不到包 | 先查物理链路、网卡、FPGA 目的 MAC/IP/端口 |
| `tcpdump` 能抓到，程序收不到 | 查端口监听、防火墙、目的端口、校验和 |
| 程序收得到但画面错 | 查分辨率、包序号、RGB565 字节序、帧头、ROI 包 |
| 画面正常但丢帧 | 查发送速率、Linux 接收缓冲、程序性能、日志打印 |

## 10. 最小抓包结论模板

联调记录可以按下面格式写：

```text
日期:
Linux 网卡:
Linux IP:
FPGA IP:
FPGA MAC:
UDP 端口:
分辨率:

tcpdump 是否抓到 UDP:
第 1 包包序号:
第 1 包帧头:
程序是否监听 1234:
程序是否显示画面:
问题现象:
结论:
```


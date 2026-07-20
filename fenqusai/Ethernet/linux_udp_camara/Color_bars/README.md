# FPGA Color Bars UDP Receiver

本目录用于在 Linux/RK 端接收 FPGA 通过以太网发送的彩条图像，并把 FPGA 的 RGB565 数据转换成 OpenCV 可显示的 BGR888 图像。

## 当前协议

| 项目 | 值 |
| --- | --- |
| FPGA 源 IP | `192.168.1.11` |
| RK/Linux IP | `192.168.1.22` |
| UDP 端口 | `1234` |
| 分辨率 | `640x360` |
| 帧率 | `60 fps` |
| 包间距 | `35 us` |
| 每帧包数 | `360` |
| 像素格式 | `RGB565` |
| Linux 显示格式 | `BGR888` |
| 字节序 | 新版 FPGA 按大端/网络字节序发送 |

FPGA 侧图像来源：

```text
1080p 彩条
-> 双线性插值缩放为 540p
-> 2x2 区域裁剪
-> 每个区域裁剪为 320x180
-> 拼接为 640x360
-> 按行通过 UDP 发送
```

## UDP Payload 格式

### 第 1 包

```text
packet_num:   4 bytes, big-endian uint32, value = 1
frame_header: 4 bytes, big-endian uint32, value = 0xf05aa50f
width:        2 bytes, big-endian uint16, value = 640
height:       2 bytes, big-endian uint16, value = 360
pixels:       1280 bytes, 640 个 RGB565 像素, big-endian uint16
```

第 1 包总 payload 长度：

```text
4 + 4 + 4 + 1280 = 1292 bytes
```

### 第 2-360 包

```text
packet_num: 4 bytes, big-endian uint32, value = 2..360
pixels:     1280 bytes, 640 个 RGB565 像素, big-endian uint16
```

后续包总 payload 长度：

```text
4 + 1280 = 1284 bytes
```

## 组帧方式

Linux 端不按“收到的顺序”拼图，而是按 FPGA 包号放到对应行。

```text
packet 1   -> row 0
packet 2   -> row 1
packet 3   -> row 2
...
packet 360 -> row 359
```

收到第 1 包并且校验帧头、宽高正确后，清空上一帧缓存并开始新帧。收到 360 行后，将整帧 RGB565 转换成 BGR888 并显示。

## 字节序说明

FPGA 新版发送顺序：

```verilog
2'd0: gmii_txd <= tx_data[31:24];
2'd1: gmii_txd <= tx_data[23:16];
2'd2: gmii_txd <= tx_data[15:8];
2'd3: gmii_txd <= tx_data[7:0];
```

因此 Linux 端按标准大端解析：

```python
packet_num = int.from_bytes(data[0:4], "big")
pixels = np.frombuffer(payload, dtype=">u2")
```

RGB565 转 BGR888：

```text
R = bit[15:11]
G = bit[10:5]
B = bit[4:0]
```

OpenCV 显示用 BGR 顺序。

## 运行

进入目录：

```bash
cd fenqusai/Ethernet/linux_udp_camara/Color_bars
```

运行接收显示程序：

```bash
python3 color_bars_udp_receiver.py
```

默认会监听：

```text
0.0.0.0:1234
```

并且只接收 FPGA 源 IP：

```text
192.168.1.11
```

退出窗口：

```text
按 q 或 ESC
```

## 常用参数

不按源 IP 过滤，接收所有发往 UDP 1234 的包：

```bash
python3 color_bars_udp_receiver.py --no-filter
```

指定 FPGA IP：

```bash
python3 color_bars_udp_receiver.py --fpga-ip 192.168.1.11
```

只接收并打印统计，不打开图像窗口：

```bash
python3 color_bars_udp_receiver.py --no-display
```

指定监听 IP 和端口：

```bash
python3 color_bars_udp_receiver.py --local-ip 0.0.0.0 --port 1234
```

## 抓包检查

确认 FPGA 是否有 UDP 包到达 RK：

```bash
sudo tcpdump -i eth1 -nn udp port 1234
```

只看 FPGA 源 IP：

```bash
sudo tcpdump -i eth1 -nn src host 192.168.1.11 and udp
```

显示 payload 十六进制，检查包号和帧头：

```bash
sudo tcpdump -i eth1 -nn -vv -X src host 192.168.1.11 and udp port 1234 -c 5
```

第 1 包 payload 开头应该类似：

```text
00 00 00 01 f0 5a a5 0f 02 80 01 68
```

含义：

```text
00 00 00 01 -> packet_num = 1
f0 5a a5 0f -> frame_header = 0xf05aa50f
02 80       -> width = 640
01 68       -> height = 360
```

## 排查

如果程序没有任何输出帧：

- 先用 `tcpdump` 确认 Linux 能抓到 UDP 包。
- 确认 RK 网卡 IP 是 `192.168.1.22/24`。
- 确认 FPGA 目的 IP 是 `192.168.1.22`。
- 确认 FPGA 目的 MAC 是 RK `eth1` 的 MAC。
- 确认 FPGA 源 IP 是 `192.168.1.11`。
- 确认 UDP 目的端口是 `1234`。
- 用 `--no-filter` 排除源 IP 不一致的问题。

如果能抓包但程序报 `First packet header mismatch`：

- 检查 FPGA 是否仍在使用旧的小端/半字节序发送方式。
- 检查第 1 包是否包含 `packet_num + frame_header + width + height + row0`。
- 检查宽高是否为 `640x360`。

如果画面颜色不对：

- 检查 RGB565 的字节序，当前代码使用 `dtype=">u2"`。
- 检查 FPGA 是否按 `tx_data[31:24] -> [23:16] -> [15:8] -> [7:0]` 发送。
- 如果 FPGA 改回 16bit 小端发送，Linux 端像素解析需要改成 `dtype="<u2"`。

如果画面偶尔卡顿或丢行：

- 增大 Linux socket 接收缓冲。
- 减少终端打印。
- 确认包间距 `35 us` 是否稳定。
- 用 C++ 接收或批量收包优化性能。


import socket
import threading

# ── 网络配置（对应 FPGA 工程参数）──────────────────────────
FPGA_IP   = "192.168.1.11"   # FPGA LOCAL_IP
FPGA_PORT = 0x8080           # FPGA LOCL_PORT

RK_IP     = "192.168.1.22"       # 监听本机所有网卡
RK_PORT   = 0x8080           # RK 端监听端口（FPGA 会往这里发）

BUFFER_SIZE = 1024
# ────────────────────────────────────────────────────────────


def receiver():
    """持续接收 FPGA 发来的 UDP 数据"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((RK_IP, RK_PORT))
    print(f"[RX] 监听 {RK_IP}:{RK_PORT} ...")

    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE)
        # FPGA 每秒发一次 "www.meyesemi.com   \n"
        print(f"[RX] 来自 {addr}  长度={len(data)}  内容: {data}")


def send(payload: bytes):
    """向 FPGA 发送一包 UDP 数据"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(payload, (FPGA_IP, FPGA_PORT))
    sock.close()
    print(f"[TX] 已发送到 {FPGA_IP}:{FPGA_PORT}  内容: {payload}")


def led_on():
    """控制 FPGA LED 亮（需 FPGA 端配合解析 udp_rec_rdata）"""
    send(bytes([0x01]))


def led_off():
    """控制 FPGA LED 灭"""
    send(bytes([0x00]))


if __name__ == "__main__":
    # 启动接收线程（后台持续收 FPGA 数据）
    t = threading.Thread(target=receiver, daemon=True)
    t.start()

    # 简单交互示例
    print("命令: on=亮灯  off=灭灯  send=自定义  q=退出")
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "on":
            led_on()
        elif cmd == "off":
            led_off()
        elif cmd == "send":
            raw = input("输入十六进制字节(如 01 02 03): ")
            try:
                payload = bytes.fromhex(raw.replace(" ", ""))
                send(payload)
            except ValueError:
                print("格式错误，请输入十六进制")
        elif cmd == "q":
            break

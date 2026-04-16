#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK端UDP通信程序
功能：
1. 接收FPGA发送的20字节测试数据（全为"w"字符）
2. 发送数据控制FPGA的LED灯

LED映射关系：
- bit0 → LED8 (0x01)
- bit1 → LED7 (0x02)
- bit2 → LED6 (0x04)
- bit3 → LED5 (0x08)
- bit4 → LED4 (0x10)
- bit5 → LED3 (0x20)
- bit6 → LED2 (0x40)
- LED1是心跳灯，由FPGA内部控制

通信参数：
- FPGA IP: 192.168.1.11
- FPGA端口: 8080
- RK IP: 192.168.1.22
- RK端口: 8080
"""

import socket
import threading
import time

# 通信参数
FPGA_IP = '192.168.1.100'
FPGA_PORT = 8080
RK_IP = '192.168.1.102'
RK_PORT = 8080

# LED映射关系
LED_MAPPING = {
    'LED2': 0x40,  # bit6
    'LED3': 0x20,  # bit5
    'LED4': 0x10,  # bit4
    'LED5': 0x08,  # bit3
    'LED6': 0x04,  # bit2
    'LED7': 0x02,  # bit1
    'LED8': 0x01   # bit0
}

# 全局变量
running = True


def receive_data():
    """接收FPGA发送的数据"""
    global running
    
    # 创建UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((RK_IP, RK_PORT))
    
    print("开始接收FPGA数据...")
    print(f"监听地址: {RK_IP}:{RK_PORT}")
    
    try:
        while running:
            # 接收数据，缓冲区大小为1024字节
            data, addr = sock.recvfrom(1024)
            if data:
                print(f"\n接收到来自 {addr} 的数据:")
                print(f"数据长度: {len(data)} 字节")
                print(f"数据内容: {data}")
                print(f"数据文本: {data.decode('utf-8', errors='ignore')}")
                
                # 检查是否为测试数据（全为'w'字符）
                if all(b == ord('w') for b in data):
                    print("✓ 确认收到FPGA测试数据（全为'w'字符）")
                
                print("-" * 50)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def control_led():
    """控制FPGA的LED灯"""
    global running
    
    # 创建UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print("\nLED控制功能")
    print("输入命令控制LED灯：")
    print("  on <LED编号> - 打开指定LED")
    print("  off <LED编号> - 关闭指定LED")
    print("  all on - 打开所有LED")
    print("  all off - 关闭所有LED")
    print("  status - 显示LED映射关系")
    print("  exit - 退出程序")
    print("-" * 50)
    
    # 当前LED状态
    current_state = 0
    
    try:
        while running:
            command = input("请输入命令: ").strip().lower()
            
            if command == 'exit':
                running = False
                break
            elif command == 'status':
                print("\nLED映射关系:")
                for led, value in LED_MAPPING.items():
                    print(f"  {led}: 0x{value:02X} (bit{bin(value).count('1')-1})")
                print(f"\n当前状态: 0x{current_state:02X}")
                
                # 显示当前LED状态
                print("当前LED状态:")
                for led, value in LED_MAPPING.items():
                    status = "开" if current_state & value else "关"
                    print(f"  {led}: {status}")
                print("-" * 50)
                
            elif command.startswith('all on'):
                # 打开所有LED (LED2-LED8)
                current_state = 0x7F
                sock.sendto(bytes([current_state]), (FPGA_IP, FPGA_PORT))
                print(f"已发送命令: 打开所有LED (0x{current_state:02X})")
                print("-" * 50)
                
            elif command.startswith('all off'):
                # 关闭所有LED (LED2-LED8)
                current_state = 0x00
                sock.sendto(bytes([current_state]), (FPGA_IP, FPGA_PORT))
                print(f"已发送命令: 关闭所有LED (0x{current_state:02X})")
                print("-" * 50)
                
            elif command.startswith('on '):
                # 打开指定LED
                led_name = command.split(' ', 1)[1].upper()
                if led_name in LED_MAPPING:
                    current_state |= LED_MAPPING[led_name]
                    sock.sendto(bytes([current_state]), (FPGA_IP, FPGA_PORT))
                    print(f"已发送命令: 打开 {led_name} (0x{current_state:02X})")
                    print("-" * 50)
                else:
                    print(f"错误: 无效的LED编号 '{led_name}'")
                    print("可用的LED编号: LED2-LED8")
                    print("-" * 50)
                    
            elif command.startswith('off '):
                # 关闭指定LED
                led_name = command.split(' ', 1)[1].upper()
                if led_name in LED_MAPPING:
                    current_state &= ~LED_MAPPING[led_name]
                    sock.sendto(bytes([current_state]), (FPGA_IP, FPGA_PORT))
                    print(f"已发送命令: 关闭 {led_name} (0x{current_state:02X})")
                    print("-" * 50)
                else:
                    print(f"错误: 无效的LED编号 '{led_name}'")
                    print("可用的LED编号: LED2-LED8")
                    print("-" * 50)
                    
            else:
                print("错误: 无效的命令")
                print("请输入有效的命令，例如: on LED8")
                print("-" * 50)
                
    except KeyboardInterrupt:
        running = False
    finally:
        sock.close()


def main():
    """主函数"""
    print("=" * 60)
    print("RK端UDP通信程序")
    print("=" * 60)
    print(f"FPGA地址: {FPGA_IP}:{FPGA_PORT}")
    print(f"RK地址: {RK_IP}:{RK_PORT}")
    print("=" * 60)
    
    # 创建并启动接收线程
    receive_thread = threading.Thread(target=receive_data)
    receive_thread.daemon = True
    receive_thread.start()
    
    # 启动LED控制功能
    control_led()
    
    # 等待接收线程结束
    receive_thread.join(timeout=1)
    
    print("\n程序已退出")


if __name__ == "__main__":
    main()
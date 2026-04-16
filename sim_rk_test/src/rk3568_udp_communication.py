#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK3568与FPGA UDP通信程序
FPGA端配置:
  - FPGA IP: 192.168.1.110
  - FPGA端口: 8080
  - RK3568 IP: 192.168.1.105
  - RK3568端口: 8080
"""

import socket
import threading
import time
from datetime import datetime

class RK3568_UDP_Communication:
    def __init__(self, local_ip='192.168.137.116', local_port=8080, 
                 fpga_ip='192.168.137.1', fpga_port=8080):
        """
        初始化UDP通信
        
        Args:
            local_ip: 本机IP地址
            local_port: 本机监听端口
            fpga_ip: FPGA的IP地址
            fpga_port: FPGA的端口
        """
        self.local_ip = local_ip
        self.local_port = local_port
        self.fpga_ip = fpga_ip
        self.fpga_port = fpga_port
        
        # 创建UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 绑定本地地址和端口
        self.sock.bind((self.local_ip, self.local_port))
        
        # 设置超时，避免阻塞
        self.sock.settimeout(1.0)
        
        self.running = False
        self.receive_thread = None
        
        print(f"[INFO] UDP Socket初始化成功")
        print(f"[INFO] 本机地址: {self.local_ip}:{self.local_port}")
        print(f"[INFO] FPGA地址: {self.fpga_ip}:{self.fpga_port}")
    
    def start_receive_thread(self):
        """启动接收线程"""
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        print("[INFO] 接收线程已启动")
    
    def _receive_loop(self):
        """接收数据循环"""
        while self.running:
            try:
                # 接收数据
                data, addr = self.sock.recvfrom(1024)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                print(f"\n[{timestamp}] 收到来自 {addr[0]}:{addr[1]} 的数据")
                print(f"[RECV] 数据长度: {len(data)} 字节")
                print(f"[RECV] 数据内容 (hex): {data.hex()}")
                print(f"[RECV] 数据内容 (ascii): {data.decode('ascii', errors='replace')}")
                print(f"[RECV] 数据内容 (bytes): {list(data)}")
                
            except socket.timeout:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                print(f"[ERROR] 接收数据错误: {e}")
                time.sleep(0.1)
    
    def send_data(self, data):
        """
        发送数据到FPGA
        
        Args:
            data: 要发送的数据，可以是字符串或字节
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            self.sock.sendto(data, (self.fpga_ip, self.fpga_port))
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] 发送数据到 {self.fpga_ip}:{self.fpga_port}")
            print(f"[SEND] 数据长度: {len(data)} 字节")
            print(f"[SEND] 数据内容 (hex): {data.hex()}")
            return True
        except Exception as e:
            print(f"[ERROR] 发送数据错误: {e}")
            return False
    
    def send_command(self, command):
        """
        发送命令到FPGA
        
        Args:
            command: 命令字符串
        """
        return self.send_data(command)
    
    def stop(self):
        """停止通信"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=2)
        self.sock.close()
        print("[INFO] UDP通信已停止")


def main():
    """主函数 - 交互式测试"""
    print("=" * 60)
    print("RK3568与FPGA UDP通信测试程序")
    print("=" * 60)
    
    # 创建通信对象
    udp_comm = RK3568_UDP_Communication()
    
    # 启动接收线程
    udp_comm.start_receive_thread()
    
    print("\n[提示] 等待FPGA发送数据...")
    print("[提示] 输入 'help' 查看命令列表")
    print("-" * 60)
    
    try:
        while True:
            # 获取用户输入
            user_input = input("\n请输入要发送的数据 (或输入 'quit' 退出): ").strip()
            
            if user_input.lower() == 'quit':
                print("[INFO] 正在退出程序...")
                break
            elif user_input.lower() == 'help':
                print("\n命令列表:")
                print("  help     - 显示帮助信息")
                print("  quit     - 退出程序")
                print("  test     - 发送测试数据")
                print("  <数据>   - 发送任意数据到FPGA")
                print("\n当前状态:")
                print(f"  本机: {udp_comm.local_ip}:{udp_comm.local_port}")
                print(f"  FPGA: {udp_comm.fpga_ip}:{udp_comm.fpga_port}")
            elif user_input.lower() == 'test':
                # 发送测试数据
                test_data = "RK3568_TEST_DATA"
                udp_comm.send_data(test_data)
            elif user_input:
                # 发送用户输入的数据
                udp_comm.send_data(user_input)
    
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，正在退出...")
    finally:
        udp_comm.stop()


if __name__ == "__main__":
    main()
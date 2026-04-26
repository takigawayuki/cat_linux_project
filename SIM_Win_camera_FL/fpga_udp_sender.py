import socket
import struct
import time
import cv2

# 配置参数
# TARGET_IP = '127.0.0.1'  # 目标IP地址，本地测试用
TARGET_IP = '192.168.137.5'  # RK端IP地址
TARGET_PORT = 1234  # 目标端口，与RK端接收端口一致
IMG_WIDTH = 640  # 图像宽度
IMG_HEIGHT = 480  # 图像高度
PACKET_COUNT = 481  # 总数据包数
FRAME_HEAD = 0xF05AA50F  # 帧头
FPS = 60  # 帧率

# 固定ROI坐标（居中矩形框）
ROI_X1 = 160  # 左上角X坐标
ROI_Y1 = 120  # 左上角Y坐标
ROI_X2 = 480  # 右下角X坐标
ROI_Y2 = 360  # 右下角Y坐标

# 创建UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

# 从摄像头获取图像数据
def generate_image_data(row, frame):
    """从摄像头获取一行图像数据，转换为RGB565格式"""
    data = b''
    # 提取指定行的数据
    row_data = frame[row, :, :]
    
    # 转换为RGB565格式
    for pixel in row_data:
        b, g, r = pixel
        r = r >> 3  # 8位转5位
        g = g >> 2  # 8位转6位
        b = b >> 3  # 8位转5位
        rgb565 = (r << 11) | (g << 5) | b
        data += struct.pack('>H', rgb565)  # 大端字节序
    return data

# 生成固定ROI坐标数据
def generate_roi_data():
    """返回固定的居中矩形框坐标"""
    return ROI_X1, ROI_Y1, ROI_X2, ROI_Y2

# 发送数据包
def send_packets():
    """发送一帧数据的所有数据包"""
    # 从摄像头获取帧
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        return
    
    # 生成ROI坐标
    x1, y1, x2, y2 = generate_roi_data()
    print(f"Sending frame with ROI: ({x1}, {y1}) - ({x2}, {y2})")
    
    # 发送第1个包（包含帧头和分辨率）
    packet_num = 1
    data = struct.pack('>I', packet_num)  # 4字节包数
    data += struct.pack('>I', FRAME_HEAD)  # 4字节帧头
    data += struct.pack('>HH', IMG_WIDTH, IMG_HEIGHT)  # 4字节分辨率
    data += generate_image_data(0, frame)  # 图像数据
    sock.sendto(data, (TARGET_IP, TARGET_PORT))
    #print(f"Sent packet {packet_num}, size: {len(data)} bytes")
    
    # 发送中间包（2-480）
    for packet_num in range(2, 481):
        row = packet_num - 1
        data = struct.pack('>I', packet_num)  # 4字节包数
        data += generate_image_data(row, frame)  # 图像数据
        sock.sendto(data, (TARGET_IP, TARGET_PORT))
        #print(f"Sent packet {packet_num}, size: {len(data)} bytes")
    
    # 发送最后一个包（481，包含坐标信息）
    packet_num = 481
    data = struct.pack('>I', packet_num)  # 4字节包数
    data += struct.pack('>HHHH', x1, y1, x2, y2)  # 8字节坐标
    # 总大小为12字节，不需要填充
    sock.sendto(data, (TARGET_IP, TARGET_PORT))
    # 最后一个包后不需要延迟，因为下一帧开始前会有帧率控制
    #print(f"Sent packet {packet_num}, size: {len(data)} bytes")

# 主循环
def main():
    try:
        frame_count = 0
        while True:
            start_time = time.time()
            send_packets()
            frame_count += 1
            print(f"Sent frame {frame_count}")
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 / FPS - elapsed)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()  # 释放摄像头
        sock.close()    # 关闭socket

if __name__ == "__main__":
    main()

#include <iostream>
#include <cstring>
#include <vector>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace std;
using namespace cv;

// 定义常量
const int UDP_PORT = 1234;
const int IMG_WIDTH = 640;
const int IMG_HEIGHT = 480;
const int PACKET_COUNT = 481; // 480行图像数据 + 1行坐标数据
const int MAX_PACKET_SIZE = 1500; // 以太网MTU

// 全局变量
Mat frame(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, Scalar(0, 0, 0));
Mat display_frame;
int x1 = 0, y1_coord = 0, x2 = IMG_WIDTH, y2 = IMG_HEIGHT;
mutex frame_mutex;
mutex roi_mutex;
bool running = true;

// 解析数据包
void parse_packet(const char* data, int size, int packet_num) {
    if (size < 4) return;
    
    // 解析包数
    int pkt_num = ntohl(*(int*)data);
    cout << "Received packet: " << pkt_num << endl;
    
    if (pkt_num == 1) {
        // 第一包：4字节包数 + 4字节帧头 + 4字节像素分辨率 + 图像数据
        if (size >= 12) {
            // 帧头（暂时忽略）
            // 分辨率（暂时忽略，使用预定义值）
            
            // 图像数据
            int data_start = 12;
            int data_len = size - data_start;
            int pixels_per_packet = data_len / 2; // 16位像素
            
            frame_mutex.lock();
            int row = 0;
            for (int i = 0; i < pixels_per_packet && row < IMG_WIDTH; i++) {
                if (data_start + i * 2 + 1 < size) {
                    uint16_t pixel = ntohs(*(uint16_t*)(data + data_start + i * 2));
                    // 16位转RGB（假设是RGB565格式）
                    uchar r = ((pixel >> 11) & 0x1F) << 3;
                    uchar g = ((pixel >> 5) & 0x3F) << 2;
                    uchar b = (pixel & 0x1F) << 3;
                    frame.at<Vec3b>(row, i) = Vec3b(b, g, r);
                }
            }
            frame_mutex.unlock();
        }
    } else if (pkt_num >= 2 && pkt_num <= 480) {
        // 中间包：4字节包数 + 图像数据
        if (size >= 4) {
            int data_start = 4;
            int data_len = size - data_start;
            int pixels_per_packet = data_len / 2; // 16位像素
            
            frame_mutex.lock();
            int row = pkt_num - 1;
            for (int i = 0; i < pixels_per_packet && i < IMG_WIDTH; i++) {
                if (data_start + i * 2 + 1 < size) {
                    uint16_t pixel = ntohs(*(uint16_t*)(data + data_start + i * 2));
                    // 16位转RGB（假设是RGB565格式）
                    uchar r = ((pixel >> 11) & 0x1F) << 3;
                    uchar g = ((pixel >> 5) & 0x3F) << 2;
                    uchar b = (pixel & 0x1F) << 3;
                    frame.at<Vec3b>(row, i) = Vec3b(b, g, r);
                }
            }
            frame_mutex.unlock();
        }
    } else if (pkt_num == 481) {
        // 最后一包：4字节包数 + 2字节x1 + 2字节y1 + 2字节x2 + 2字节y2
        if (size >= 12) {
            roi_mutex.lock();
            x1 = ntohs(*(uint16_t*)(data + 4));
            y1_coord = ntohs(*(uint16_t*)(data + 6));
            x2 = ntohs(*(uint16_t*)(data + 8));
            y2 = ntohs(*(uint16_t*)(data + 10));
            // 确保坐标在有效范围内
            x1 = max(0, min(x1, IMG_WIDTH - 1));
            y1_coord = max(0, min(y1_coord, IMG_HEIGHT - 1));
            x2 = max(0, min(x2, IMG_WIDTH - 1));
            y2 = max(0, min(y2, IMG_HEIGHT - 1));
            // 确保x1 < x2, y1 < y2
            if (x1 > x2) std::swap(x1, x2);
            if (y1_coord > y2) std::swap(y1_coord, y2);
            roi_mutex.unlock();
            cout << "Received ROI: (" << x1 << "," << y1_coord << ") - (" << x2 << "," << y2 << ")" << endl;
        }
    }
}

// UDP接收线程
void udp_receive_thread() {
    int sockfd;
    struct sockaddr_in servaddr, cliaddr;
    char buffer[MAX_PACKET_SIZE];
    
    // 创建UDP套接字
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        return;
    }
    
    // 设置服务器地址
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(UDP_PORT);
    
    // 绑定端口
    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        perror("bind failed");
        close(sockfd);
        return;
    }
    
    cout << "UDP server started on port " << UDP_PORT << endl;
    
    while (running) {
        socklen_t len = sizeof(cliaddr);
        int n = recvfrom(sockfd, (char *)buffer, MAX_PACKET_SIZE, 0, (struct sockaddr *)&cliaddr, &len);
        if (n > 0) {
            parse_packet(buffer, n, 0);
        }
    }
    
    close(sockfd);
}

// 显示线程
void display_thread() {
    namedWindow("FPGA Camera", WINDOW_NORMAL);
    resizeWindow("FPGA Camera", 800, 600);
    
    while (running) {
        frame_mutex.lock();
        Mat temp_frame = frame.clone();
        frame_mutex.unlock();
        
        roi_mutex.lock();
        int roi_x1 = x1, roi_y1 = y1_coord, roi_x2 = x2, roi_y2 = y2;
        roi_mutex.unlock();
        
        // 绘制ROI矩形
        rectangle(temp_frame, Point(roi_x1, roi_y1), Point(roi_x2, roi_y2), Scalar(0, 255, 0), 2);
        
        // 提取ROI区域
        if (roi_x2 > roi_x1 && roi_y2 > roi_y1) {
            display_frame = temp_frame(Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));
            if (!display_frame.empty()) {
                imshow("FPGA Camera", display_frame);
            }
        } else {
            imshow("FPGA Camera", temp_frame);
        }
        
        // 按ESC键退出
        if (waitKey(1) == 27) {
            running = false;
        }
    }
    
    destroyAllWindows();
}

int main() {
    // 启动UDP接收线程
    thread receive_thread(udp_receive_thread);
    
    // 启动显示线程
    thread display_thread_func(display_thread);
    
    // 等待线程结束
    receive_thread.join();
    display_thread_func.join();
    
    return 0;
}

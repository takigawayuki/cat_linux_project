#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <atomic>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define UDP_PORT 1234
#define MAX_PACKET_SIZE 2048

#define IMG_WIDTH 640
#define IMG_HEIGHT 480

#define TOTAL_PACKETS 961

// ================= 全局 =================
static uint8_t image_rgb565[IMG_WIDTH * IMG_HEIGHT * 2] = {0};

static atomic<int> g_x1(0), g_y1(0), g_x2(IMG_WIDTH), g_y2(IMG_HEIGHT);
static atomic<bool> g_new_frame(false);

// ================= RGB565 -> RGB888 =================
void rgb565_to_rgb888(const uint8_t* src, uint8_t* dst)
{
    int pixels = IMG_WIDTH * IMG_HEIGHT;

    for (int i = 0, j = 0; i < pixels * 2; i += 2, j += 3)
    {
        uint16_t p = *(uint16_t*)(src + i);

        uint8_t r = (p >> 11) & 0x1F;
        uint8_t g = (p >> 5) & 0x3F;
        uint8_t b = p & 0x1F;

        dst[j]     = (r << 3) | (r >> 2);
        dst[j + 1] = (g << 2) | (g >> 4);
        dst[j + 2] = (b << 3) | (b >> 2);
    }
}

// ================= 接收线程 =================
void recv_thread()
{
    int sockfd;
    sockaddr_in addr{}, client{};
    socklen_t len = sizeof(client);

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    addr.sin_family = AF_INET;
    addr.sin_port = htons(UDP_PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(sockfd, (sockaddr*)&addr, sizeof(addr));

    cout << "UDP started (half-line mode)" << endl;

    uint8_t buffer[MAX_PACKET_SIZE];

    int packet_cnt = 0;

    while (true)
    {
        int recv_len = recvfrom(sockfd, buffer, MAX_PACKET_SIZE, 0,
                                (sockaddr*)&client, &len);

        if (recv_len <= 0) continue;

        uint32_t pkt = ntohl(*(uint32_t*)buffer);

        // ===== ROI包 =====
        if (pkt == TOTAL_PACKETS)
        {
            if (recv_len >= 12)
            {
                g_x1 = ntohs(*(uint16_t*)(buffer + 4));
                g_y1 = ntohs(*(uint16_t*)(buffer + 6));
                g_x2 = ntohs(*(uint16_t*)(buffer + 8));
                g_y2 = ntohs(*(uint16_t*)(buffer + 10));

                g_new_frame = true;
            }
            continue;
        }

        // ===== 图像包 =====
        uint8_t* payload;
        int payload_len;

        if (pkt == 1)
        {
            payload = buffer + 16;   // 跳过帧头+分辨率
            payload_len = recv_len - 16;
        }
        else
        {
            payload = buffer + 4;
            payload_len = recv_len - 4;
        }

        // ===== 半行解析 =====
        int row = (pkt - 1) / 2;
        bool first_half = ((pkt - 1) % 2 == 0);

        if (row < 0 || row >= IMG_HEIGHT) continue;

        int offset;

        if (first_half)
            offset = row * IMG_WIDTH * 2;
        else
            offset = row * IMG_WIDTH * 2 + (IMG_WIDTH / 2) * 2;

        if (offset + payload_len <= IMG_WIDTH * IMG_HEIGHT * 2)
        {
            memcpy(image_rgb565 + offset, payload, payload_len);
        }

        // ===== 防卡死：周期刷新 =====
        packet_cnt++;
        if (packet_cnt >= 200)
        {
            g_new_frame = true;
            packet_cnt = 0;
        }
    }
}

// ================= 显示线程 =================
void display_thread()
{
    Mat rgb(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);

    while (true)
    {
        if (!g_new_frame)
        {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        g_new_frame = false;

        rgb565_to_rgb888(image_rgb565, rgb.data);

        int x1 = g_x1, y1 = g_y1, x2 = g_x2, y2 = g_y2;

        x1 = max(0, min(x1, IMG_WIDTH - 1));
        x2 = max(0, min(x2, IMG_WIDTH - 1));
        y1 = max(0, min(y1, IMG_HEIGHT - 1));
        y2 = max(0, min(y2, IMG_HEIGHT - 1));

        if (x2 > x1 && y2 > y1)
        {
            Rect roi(x1, y1, x2 - x1, y2 - y1);
            Mat cropped = rgb(roi);
            imshow("ROI", cropped);
        }
        else
        {
            imshow("ROI", rgb);
        }

        waitKey(1);
    }
}

// ================= main =================
int main()
{
    cout << "Start receiver..." << endl;

    thread t1(recv_thread);
    thread t2(display_thread);

    t1.join();
    t2.join();

    return 0;
}
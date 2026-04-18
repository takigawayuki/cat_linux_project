#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include <sys/socket.h>
#include <linux/socket.h>
#include <opencv2/opencv.hpp>

#define PORT 1234
#define WIDTH 640
#define HEIGHT 480
#define TOTAL_PACKETS 481
#define BATCH 64

int main()
{
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    int buf_size = 64 * 1024 * 1024;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));

    std::cout << "🚀 UDP接收启动: " << PORT << std::endl;

    // ===== 批量接收 =====
    std::vector<std::vector<uint8_t>> buffers(BATCH, std::vector<uint8_t>(2048));
    std::vector<iovec> iov(BATCH);
    std::vector<mmsghdr> msgs(BATCH);

    for (int i = 0; i < BATCH; i++)
    {
        iov[i].iov_base = buffers[i].data();
        iov[i].iov_len = buffers[i].size();

        msgs[i].msg_hdr.msg_iov = &iov[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
    }

    cv::Mat image(HEIGHT, WIDTH, CV_8UC3);

    // 👉 记录包
    std::vector<uint8_t> received(TOTAL_PACKETS + 5, 0);

    int x1=0,y1=0,x2=0,y2=0;

    int frame_cnt = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    while (true)
    {
        int ret = recvmmsg(sockfd, msgs.data(), BATCH, 0, NULL);
        if (ret <= 0) continue;

        for (int m = 0; m < ret; m++)
        {
            uint8_t* buffer = buffers[m].data();
            int len = msgs[m].msg_len;

            if (len < 4) continue;

            uint32_t packet_num = ntohl(*(uint32_t*)buffer);
            uint8_t* payload = buffer + 4;
            int payload_len = len - 4;

            if (packet_num > TOTAL_PACKETS) continue;

            // 👉 标记收到
            received[packet_num] = 1;

            // ===== ROI包（帧结束）=====
            if (payload_len <= 16)
            {
                if (payload_len >= 8)
                {
                    x1 = ntohs(*(uint16_t*)(payload));
                    y1 = ntohs(*(uint16_t*)(payload+2));
                    x2 = ntohs(*(uint16_t*)(payload+4));
                    y2 = ntohs(*(uint16_t*)(payload+6));
                }

                // ===== 统计包 =====
                int count = 0;
                for (int i = 1; i <= TOTAL_PACKETS; i++)
                    count += received[i];

                frame_cnt++;

                auto now = std::chrono::high_resolution_clock::now();
                double fps = frame_cnt /
                    std::chrono::duration<double>(now - last_time).count();

                std::cout << "Frame OK: " << count
                          << "/" << TOTAL_PACKETS
                          << "  FPS: " << fps << std::endl;

                // ===== 显示（不要求满帧）=====
                int xmin = std::min(x1,x2);
                int xmax = std::max(x1,x2);
                int ymin = std::min(y1,y2);
                int ymax = std::max(y1,y2);

                xmin = std::max(0, std::min(xmin, WIDTH-1));
                xmax = std::max(0, std::min(xmax, WIDTH-1));
                ymin = std::max(0, std::min(ymin, HEIGHT-1));
                ymax = std::max(0, std::min(ymax, HEIGHT-1));

                if (xmax > xmin && ymax > ymin)
                {
                    cv::Mat crop = image(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
                    cv::resize(crop, crop, cv::Size(WIDTH, HEIGHT));
                    cv::imshow("FPGA", crop);
                }
                else
                {
                    cv::imshow("FPGA", image);
                }

                cv::waitKey(1);

                // 👉 重置
                std::fill(received.begin(), received.end(), 0);

                continue;
            }

            // ===== 图像数据 =====
            int row = packet_num - 1;
            if (row < 0 || row >= HEIGHT) continue;

            uint16_t* pixels = (uint16_t*)payload;
            int pixel_count = payload_len / 2;

            cv::Vec3b* row_ptr = image.ptr<cv::Vec3b>(row);

            for (int i = 0; i < pixel_count && i < WIDTH; i++)
            {
                uint16_t p = ntohs(pixels[i]);

                uint8_t r = ((p >> 11) & 0x1F) << 3;
                uint8_t g = ((p >> 5) & 0x3F) << 2;
                uint8_t b = (p & 0x1F) << 3;

                row_ptr[i][0] = b;
                row_ptr[i][1] = g;
                row_ptr[i][2] = r;
            }
        }
    }

    close(sockfd);
    return 0;
}
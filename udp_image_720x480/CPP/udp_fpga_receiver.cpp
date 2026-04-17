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
#define WIDTH 720
#define HEIGHT 480
#define FRAME_HEADER 0xf05aa50f

#define BATCH_SIZE 32   // 每次收32个包（关键优化）

int main()
{
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    int buf_size = 64 * 1024 * 1024;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));

    int busy_poll = 50;
    setsockopt(sockfd, SOL_SOCKET, SO_BUSY_POLL, &busy_poll, sizeof(busy_poll));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));

    std::cout << "🚀 高性能UDP接收启动: " << PORT << std::endl;

    // 批量接收结构
    std::vector<std::vector<uint8_t>> buffers(BATCH_SIZE, std::vector<uint8_t>(2048));
    std::vector<iovec> iovecs(BATCH_SIZE);
    std::vector<mmsghdr> msgs(BATCH_SIZE);

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        iovecs[i].iov_base = buffers[i].data();
        iovecs[i].iov_len = buffers[i].size();

        msgs[i].msg_hdr.msg_iov = &iovecs[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
    }

    cv::Mat image(HEIGHT, WIDTH, CV_8UC3);
    std::vector<uint8_t> received(2000, 0);

    int x1=0,y1=0,x2=0,y2=0;

    int frame_count = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    while (true)
    {
        int ret = recvmmsg(sockfd, msgs.data(), BATCH_SIZE, 0, NULL);
        if (ret <= 0) continue;

        for (int m = 0; m < ret; m++)
        {
            uint8_t* buffer = buffers[m].data();
            int len = msgs[m].msg_len;

            if (len < 4) continue;

            uint32_t packet_num = ntohl(*(uint32_t*)buffer);
            uint8_t* payload = buffer + 4;
            int payload_len = len - 4;

            // ===== 帧头 =====
            if (payload_len >= 8)
            {
                uint32_t header = ntohl(*(uint32_t*)payload);
                if (header == FRAME_HEADER)
                {
                    std::fill(received.begin(), received.end(), 0);
                    payload += 8;
                    payload_len -= 8;
                }
            }

            // ===== 最后一包 =====
            if (payload_len == 8 || payload_len == 16)
            {
                x1 = ntohs(*(uint16_t*)(payload));
                y1 = ntohs(*(uint16_t*)(payload+2));
                x2 = ntohs(*(uint16_t*)(payload+4));
                y2 = ntohs(*(uint16_t*)(payload+6));

                int total_expected = packet_num - 1;

                int count = 0;
                for(int i=1;i<=total_expected;i++)
                    count += received[i];

                if (count > total_expected * 0.9)  // 提高到90%
                {
                    frame_count++;

                    // ===== FPS计算 =====
                    if (frame_count % 30 == 0)
                    {
                        auto now = std::chrono::high_resolution_clock::now();
                        double fps = 30.0 / std::chrono::duration<double>(now - last_time).count();
                        last_time = now;

                        std::cout << "Frame OK: " << count << "/" << total_expected
                                  << " | FPS: " << fps << std::endl;
                    }

                    int xmin = std::min(x1,x2);
                    int xmax = std::max(x1,x2);
                    int ymin = std::min(y1,y2);
                    int ymax = std::max(y1,y2);

                    xmin = std::max(0, std::min(xmin, WIDTH-1));
                    xmax = std::max(0, std::min(xmax, WIDTH-1));
                    ymin = std::max(0, std::min(ymin, HEIGHT-1));
                    ymax = std::max(0, std::min(ymax, HEIGHT-1));

                    static int show_cnt = 0;
                    show_cnt++;

                    if (show_cnt % 2 == 0)  // 控制显示频率
                    {
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
                    }
                }
                continue;
            }

            // ===== 图像包 =====
            if (packet_num >= received.size()) continue;
            if (received[packet_num]) continue;

            received[packet_num] = 1;

            int row = (packet_num - 1) / 2;
            int col_start = ((packet_num - 1) % 2) * (WIDTH / 2);

            uint16_t* pixels = (uint16_t*)payload;
            int pixel_count = payload_len / 2;

            cv::Vec3b* row_ptr = image.ptr<cv::Vec3b>(row);

            for (int i = 0; i < pixel_count; i++)
            {
                uint16_t p = ntohs(pixels[i]);

                row_ptr[col_start + i][0] = (p & 0x1F) << 3;
                row_ptr[col_start + i][1] = ((p >> 5) & 0x3F) << 2;
                row_ptr[col_start + i][2] = ((p >> 11) & 0x1F) << 3;
            }
        }
    }

    close(sockfd);
    return 0;
}
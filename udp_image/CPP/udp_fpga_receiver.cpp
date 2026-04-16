#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#define PORT 1234
#define WIDTH 720
#define HEIGHT 480
#define FRAME_HEADER 0xf05aa50f

int main()
{
    int sockfd;
    struct sockaddr_in addr;

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    int buf_size = 32 * 1024 * 1024;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));

    std::cout << "监听 UDP 端口: " << PORT << std::endl;

    std::vector<uint8_t> buffer(2048);

    cv::Mat image(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat display_img;

    std::vector<bool> received(2000, false);

    int x1=0,y1=0,x2=0,y2=0;

    int frame_cnt = 0;
    auto last_time = std::chrono::steady_clock::now();

    while (true)
    {
        ssize_t len = recvfrom(sockfd, buffer.data(), buffer.size(), 0, NULL, NULL);
        if (len < 4) continue;

        uint32_t packet_num = ntohl(*(uint32_t*)&buffer[0]);
        uint8_t* payload = buffer.data() + 4;
        int payload_len = len - 4;

        // 帧头
        if (payload_len >= 8)
        {
            uint32_t header = ntohl(*(uint32_t*)(payload));
            if (header == FRAME_HEADER)
            {
                std::fill(received.begin(), received.end(), false);
                payload += 8;
                payload_len -= 8;
            }
        }

        // 最后一包
        if (payload_len == 8 || payload_len == 16)
        {
            x1 = ntohs(*(uint16_t*)(payload));
            y1 = ntohs(*(uint16_t*)(payload+2));
            x2 = ntohs(*(uint16_t*)(payload+4));
            y2 = ntohs(*(uint16_t*)(payload+6));

            int total_expected = packet_num - 1;

            int count = 0;
            for(int i=1;i<=total_expected;i++)
                if(received[i]) count++;

            if (count > total_expected * 0.9)
            {
                frame_cnt++;

                // ROI
                int xmin = std::max(0, std::min(std::min(x1,x2), WIDTH-1));
                int xmax = std::max(0, std::min(std::max(x1,x2), WIDTH-1));
                int ymin = std::max(0, std::min(std::min(y1,y2), HEIGHT-1));
                int ymax = std::max(0, std::min(std::max(y1,y2), HEIGHT-1));

                if (xmax > xmin && ymax > ymin)
                {
                    cv::Mat crop = image(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
                    cv::resize(crop, display_img, cv::Size(WIDTH, HEIGHT));
                }
                else
                {
                    display_img = image;
                }

                // 👉 FPS统计
                auto now = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(now - last_time).count();

                if (sec >= 1.0)
                {
                    double fps = frame_cnt / sec;
                    std::cout << "FPS: " << fps << std::endl;
                    frame_cnt = 0;
                    last_time = now;
                }

                // 👉 显示（限帧）
                static int show_skip = 0;
                show_skip++;

                if (show_skip % 2 == 0)   // 👉 控制显示帧率
                {
                    cv::imshow("FPGA", display_img);
                    cv::waitKey(1);
                }
            }

            continue;
        }

        // 图像包
        if (packet_num >= received.size()) continue;
        if (received[packet_num]) continue;

        received[packet_num] = true;

        int row = (packet_num - 1) / 2;
        int col_start = ((packet_num - 1) % 2) * (WIDTH / 2);

        uint16_t* pixels = (uint16_t*)payload;
        int pixel_count = payload_len / 2;

        cv::Vec3b* row_ptr = image.ptr<cv::Vec3b>(row);

        for (int i = 0; i < pixel_count; i++)
        {
            uint16_t p = ntohs(pixels[i]);

            row_ptr[col_start + i][2] = ((p >> 11) & 0x1F) << 3;
            row_ptr[col_start + i][1] = ((p >> 5) & 0x3F) << 2;
            row_ptr[col_start + i][0] = (p & 0x1F) << 3;
        }
    }

    close(sockfd);
    return 0;
}
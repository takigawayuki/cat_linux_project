#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define PORT 1234
#define WIDTH 720
#define HEIGHT 480
#define FRAME_HEADER 0xf05aa50f

int main()
{
    int sockfd;
    struct sockaddr_in addr;

    // 创建socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    int buf_size = 16 * 1024 * 1024;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));

    std::cout << "监听 UDP 端口: " << PORT << std::endl;

    std::vector<uint8_t> buffer(2048);

    // 图像缓存
    cv::Mat image(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0,0,0));

    // 包记录
    std::vector<bool> received(2000, false);

    int x1=0,y1=0,x2=0,y2=0;

    while (true)
    {
        ssize_t len = recvfrom(sockfd, buffer.data(), buffer.size(), 0, NULL, NULL);
        if (len < 4) continue;

        uint32_t packet_num = ntohl(*(uint32_t*)&buffer[0]);
        uint8_t* payload = buffer.data() + 4;
        int payload_len = len - 4;

        // =========================
        // 帧头判断（第一包）
        // =========================
        if (payload_len >= 8)
        {
            uint32_t header = ntohl(*(uint32_t*)(payload));

            if (header == FRAME_HEADER)
            {
                // 新帧
                std::fill(received.begin(), received.end(), false);
                image.setTo(cv::Scalar(0,0,0));

                payload += 8;
                payload_len -= 8;
            }
        }

        // =========================
        // 最后一包（坐标）
        // =========================
        if (payload_len == 8 || payload_len == 16)
        {
            if (payload_len >= 8)
            {
                x1 = ntohs(*(uint16_t*)(payload));
                y1 = ntohs(*(uint16_t*)(payload+2));
                x2 = ntohs(*(uint16_t*)(payload+4));
                y2 = ntohs(*(uint16_t*)(payload+6));
            }

            int total_expected = packet_num - 1;

            int count = 0;
            for(int i=1;i<=total_expected;i++)
                if(received[i]) count++;

            if (count > total_expected * 0.8)  // 👉 容忍20%丢包
            {
                std::cout << "Frame OK: " << count << "/" << total_expected << std::endl;

                // ROI裁剪
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
            }

            continue;
        }

        // =========================
        // 图像包
        // =========================
        if (packet_num >= received.size()) continue;
        if (received[packet_num]) continue;

        received[packet_num] = true;

        int row = (packet_num - 1) / 2;
        int col_start = ((packet_num - 1) % 2) * (WIDTH / 2);

        if (payload_len % 2 != 0) continue;

        uint16_t* pixels = (uint16_t*)payload;
        int pixel_count = payload_len / 2;

        for (int i = 0; i < pixel_count; i++)
        {
            uint16_t p = ntohs(pixels[i]);

            uint8_t r = ((p >> 11) & 0x1F) << 3;
            uint8_t g = ((p >> 5) & 0x3F) << 2;
            uint8_t b = (p & 0x1F) << 3;

            int x = col_start + i;
            int y = row;

            if (x < WIDTH && y < HEIGHT)
            {
                image.at<cv::Vec3b>(y,x) = cv::Vec3b(b,g,r);
            }
        }
    }

    close(sockfd);
    return 0;
}
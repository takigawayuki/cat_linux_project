#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <sys/socket.h>
#include <opencv2/opencv.hpp>

#define PORT           1234
#define WIDTH          640
#define HEIGHT         480

#define TOTAL_PACKETS  961
#define IMAGE_PACKETS  960
#define BATCH          128

#ifndef SO_RCVBUFFORCE
#define SO_RCVBUFFORCE 33
#endif

static cv::Mat g_front(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0));
static cv::Mat g_back(HEIGHT, WIDTH,  CV_8UC3, cv::Scalar(0));
static std::mutex g_swap_mtx;
static std::atomic<bool> g_new_frame{false};

static std::atomic<int> g_x1{0}, g_y1{0}, g_x2{WIDTH}, g_y2{HEIGHT};

// ===== 接收统计 =====
static uint8_t received[TOTAL_PACKETS] = {0};
static int received_cnt = 0;
static int pkt_cnt = 0;

// ---- 显示线程 ----
void display_thread()
{
    cv::Mat show;

    while (true)
    {
        if (!g_new_frame.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        {
            std::lock_guard<std::mutex> lk(g_swap_mtx);
            g_front.copyTo(show);
            g_new_frame.store(false, std::memory_order_relaxed);
        }

        int x1 = g_x1, y1 = g_y1, x2 = g_x2, y2 = g_y2;
        int xmin = std::max(0, std::min(x1, x2));
        int xmax = std::min(WIDTH-1,  std::max(x1, x2));
        int ymin = std::max(0, std::min(y1, y2));
        int ymax = std::min(HEIGHT-1, std::max(y1, y2));

        if (xmax > xmin && ymax > ymin)
            cv::rectangle(show, cv::Point(xmin,ymin), cv::Point(xmax,ymax),
                          cv::Scalar(0,255,0), 2);

        cv::imshow("FPGA", show);
        cv::waitKey(1);
    }
}

int main()
{
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    int buf_size = 64 * 1024 * 1024;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUFFORCE, &buf_size, sizeof(buf_size)) < 0)
        setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));

    std::cout << "UDP接收启动 (最终稳定版)" << std::endl;

    std::thread disp(display_thread);
    disp.detach();

    std::vector<std::vector<uint8_t>> buffers(BATCH, std::vector<uint8_t>(2048));
    std::vector<iovec>   iov(BATCH);
    std::vector<mmsghdr> msgs(BATCH);

    for (int i = 0; i < BATCH; i++) {
        iov[i].iov_base = buffers[i].data();
        iov[i].iov_len  = buffers[i].size();
        msgs[i].msg_hdr.msg_iov    = &iov[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
    }

    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_cnt  = 0;

    while (true)
    {
        int ret = recvmmsg(sockfd, msgs.data(), BATCH, MSG_WAITFORONE, NULL);
        if (ret <= 0) continue;

        for (int m = 0; m < ret; m++)
        {
            uint8_t* buf = buffers[m].data();
            int len = msgs[m].msg_len;
            if (len < 4) continue;

            uint32_t pkt = ntohl(*(uint32_t*)buf);
            if (pkt == 0 || pkt > TOTAL_PACKETS) continue;

            uint8_t* payload = buf + 4;

            // ===== 统计 =====
            if (!received[pkt]) {
                received[pkt] = 1;
                received_cnt++;
            }

            pkt_cnt++;

            // ===== ROI包（强触发）=====
            if (pkt == TOTAL_PACKETS)
            {
                if (len >= 12) {
                    g_x1 = ntohs(*(uint16_t*)(payload + 0));
                    g_y1 = ntohs(*(uint16_t*)(payload + 2));
                    g_x2 = ntohs(*(uint16_t*)(payload + 4));
                    g_y2 = ntohs(*(uint16_t*)(payload + 6));
                }

                // ⭐ 用ROI包作为帧结束（最可靠）
                {
                    std::lock_guard<std::mutex> lk(g_swap_mtx);
                    std::swap(g_front, g_back);
                    g_new_frame.store(true, std::memory_order_relaxed);
                }

                frame_cnt++;

                auto now = std::chrono::high_resolution_clock::now();
                double fps = frame_cnt /
                    std::chrono::duration<double>(now - last_time).count();

                float ratio = received_cnt / (float)IMAGE_PACKETS;

                std::cout << "\rFrame " << frame_cnt
                          << " FPS: " << fps
                          << " recv: " << received_cnt << "/960"
                          << " (" << ratio * 100 << "%)"
                          << std::flush;

                // 清空
                memset(received, 0, sizeof(received));
                received_cnt = 0;
                pkt_cnt = 0;

                continue;
            }

            // ===== 半行解析 =====
            int row = (pkt - 1) / 2;
            bool is_first_half = ((pkt - 1) % 2 == 0);

            uint16_t* pixels;

            if (pkt == 1)
                pixels = (uint16_t*)(payload + 8);
            else
                pixels = (uint16_t*)payload;

            if (row < 0 || row >= HEIGHT) continue;

            cv::Vec3b* row_ptr = g_back.ptr<cv::Vec3b>(row);

            int start = is_first_half ? 0 : WIDTH / 2;
            int count = WIDTH / 2;

            for (int i = 0; i < count; i++) {
                uint16_t p = ntohs(pixels[i]);
                int idx = start + i;

                row_ptr[idx][0] = (p & 0x1F) << 3;
                row_ptr[idx][1] = ((p >> 5) & 0x3F) << 2;
                row_ptr[idx][2] = ((p >> 11) & 0x1F) << 3;
            }

            // ===== 兜底触发（防ROI丢失）=====
            if (pkt_cnt > 400)
            {
                {
                    std::lock_guard<std::mutex> lk(g_swap_mtx);
                    std::swap(g_front, g_back);
                    g_new_frame.store(true, std::memory_order_relaxed);
                }

                pkt_cnt = 0;
            }
        }
    }

    close(sockfd);
    return 0;
}
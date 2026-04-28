#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <sys/socket.h>
#include <opencv2/opencv.hpp>

#define PORT           1234
#define WIDTH          320
#define HEIGHT         240
#define TOTAL_PACKETS  241
#define BATCH          128

#ifndef SO_RCVBUFFORCE
#define SO_RCVBUFFORCE 33
#endif

// raw 双缓冲（接收线程写 RGB565，解码线程读）
static uint16_t g_raw[2][HEIGHT][WIDTH];
static int      g_raw_write = 0;
static std::mutex              g_raw_mtx;
static std::condition_variable g_raw_cv;
static bool                    g_raw_ready = false;

// RGB888 双缓冲（解码线程写，显示线程读）
static cv::Mat g_front(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0));
static cv::Mat g_back (HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0));
static std::mutex              g_disp_mtx;
static std::condition_variable g_disp_cv;
static bool                    g_disp_ready = false;

// ROI
static std::atomic<int> g_x1{0}, g_y1{0}, g_x2{WIDTH}, g_y2{HEIGHT};

// 采集帧率（接收线程更新，显示线程读）
static std::atomic<double> g_capture_fps{0.0};
// 显示帧率（显示线程自己统计）
static std::atomic<double> g_display_fps{0.0};

void decode_thread()
{
    while (true)
    {
        int idx;
        {
            std::unique_lock<std::mutex> lk(g_raw_mtx);
            g_raw_cv.wait(lk, []{ return g_raw_ready; });
            idx = 1 - g_raw_write;
            g_raw_ready = false;
        }

        uint16_t (*src)[WIDTH] = g_raw[idx];
        for (int row = 0; row < HEIGHT; row++) {
            cv::Vec3b* dst = g_back.ptr<cv::Vec3b>(row);
            for (int i = 0; i < WIDTH; i++) {
                uint16_t p = ntohs(src[row][i]);
                uint8_t r5 = (p >> 11) & 0x1F;
                uint8_t g6 = (p >> 5)  & 0x3F;
                uint8_t b5 = p & 0x1F;
                dst[i][2] = (r5 << 3) | (r5 >> 2);
                dst[i][1] = (g6 << 2) | (g6 >> 4);
                dst[i][0] = (b5 << 3) | (b5 >> 2);
            }
        }

        {
            std::lock_guard<std::mutex> lk(g_disp_mtx);
            std::swap(g_front, g_back);
            g_disp_ready = true;
        }
        g_disp_cv.notify_one();
    }
}

void display_thread()
{
    cv::Mat show;
    auto last_disp = std::chrono::steady_clock::now();

    while (true)
    {
        {
            std::unique_lock<std::mutex> lk(g_disp_mtx);
            g_disp_cv.wait(lk, []{ return g_disp_ready; });
            g_front.copyTo(show);
            g_disp_ready = false;
        }

        // 统计显示帧率
        auto now_disp = std::chrono::steady_clock::now();
        double disp_elapsed = std::chrono::duration<double>(now_disp - last_disp).count();
        if (disp_elapsed > 0)
            g_display_fps.store(1.0 / disp_elapsed, std::memory_order_relaxed);
        last_disp = now_disp;

        // 画 ROI 框
        int x1 = g_x1, y1 = g_y1, x2 = g_x2, y2 = g_y2;
        int xmin = std::max(0, std::min(x1, x2));
        int xmax = std::min(WIDTH-1,  std::max(x1, x2));
        int ymin = std::max(0, std::min(y1, y2));
        int ymax = std::min(HEIGHT-1, std::max(y1, y2));
        if (xmax > xmin && ymax > ymin)
            cv::rectangle(show, cv::Point(xmin,ymin), cv::Point(xmax,ymax),
                          cv::Scalar(0,255,0), 2);

        // 左上角显示采集帧率和显示帧率
        char buf[64];
        snprintf(buf, sizeof(buf), "Cap:%.1f  Disp:%.1f",
                 g_capture_fps.load(), g_display_fps.load());
        cv::putText(show, buf, cv::Point(4, 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 0, 0), 2, cv::LINE_AA);   // 黑色描边
        cv::putText(show, buf, cv::Point(4, 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 255, 255), 1, cv::LINE_AA); // 黄色字

        cv::imshow("FPGA 320x240", show);
        cv::waitKey(1);
    }
}

int main()
{
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    int buf_size = 64 * 1024 * 1024;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUFFORCE, &buf_size, sizeof(buf_size)) < 0)
        setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));

    int actual = 0; socklen_t alen = sizeof(actual);
    getsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &actual, &alen);

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));

    std::cout << "UDP接收启动: port=" << PORT
              << "  RCVBUF=" << actual / 1024 << " KB" << std::endl;

    std::thread(decode_thread).detach();
    std::thread(display_thread).detach();

    std::vector<std::vector<uint8_t>> buffers(BATCH, std::vector<uint8_t>(2048));
    std::vector<iovec>   iov(BATCH);
    std::vector<mmsghdr> msgs(BATCH);
    for (int i = 0; i < BATCH; i++) {
        iov[i].iov_base = buffers[i].data();
        iov[i].iov_len  = buffers[i].size();
        msgs[i].msg_hdr.msg_iov    = &iov[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
    }

    std::vector<uint8_t> received(HEIGHT + 2, 0);
    int rows_received = 0;
    int frame_cnt = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    long long dbg_pkts = 0;
    auto dbg_time = std::chrono::steady_clock::now();

    while (true)
    {
        int ret = recvmmsg(sockfd, msgs.data(), BATCH, MSG_WAITFORONE, NULL);
        if (ret <= 0) continue;

        dbg_pkts += ret;
        auto dbg_now = std::chrono::steady_clock::now();
        double dbg_elapsed = std::chrono::duration<double>(dbg_now - dbg_time).count();
        if (dbg_elapsed >= 3.0) {
            fprintf(stderr, "\n[dbg] %.0f pkts/s  rows=%d\n",
                    dbg_pkts / dbg_elapsed, rows_received);
            dbg_pkts = 0;
            dbg_time = dbg_now;
        }

        for (int m = 0; m < ret; m++)
        {
            uint8_t* buf = buffers[m].data();
            int len = msgs[m].msg_len;
            if (len < 4) continue;

            uint32_t pkt = ntohl(*(uint32_t*)buf);
            if (pkt == 0 || pkt > TOTAL_PACKETS) continue;

            uint8_t* payload     = buf + 4;
            int      payload_len = len - 4;

            if (pkt == TOTAL_PACKETS) {
                if (payload_len >= 8) {
                    g_x1 = ntohs(*(uint16_t*)(payload + 0));
                    g_y1 = ntohs(*(uint16_t*)(payload + 2));
                    g_x2 = ntohs(*(uint16_t*)(payload + 4));
                    g_y2 = ntohs(*(uint16_t*)(payload + 6));
                }
                continue;
            }

            int row = (int)pkt - 1;
            if (row < 0 || row >= HEIGHT) continue;

            uint16_t* pixels;
            if (pkt == 1) {
                if (payload_len < 8 + WIDTH * 2) continue;
                pixels = (uint16_t*)(payload + 8);
            } else {
                if (payload_len < WIDTH * 2) continue;
                pixels = (uint16_t*)payload;
            }

            if (!received[pkt]) {
                received[pkt] = 1;
                rows_received++;
            }

            memcpy(g_raw[g_raw_write][row], pixels, WIDTH * 2);

            if (rows_received == HEIGHT) {
                {
                    std::lock_guard<std::mutex> lk(g_raw_mtx);
                    g_raw_write = 1 - g_raw_write;
                    g_raw_ready = true;
                }
                g_raw_cv.notify_one();

                frame_cnt++;
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - last_time).count();
                double fps = 1.0 / elapsed;
                g_capture_fps.store(fps, std::memory_order_relaxed);

                std::cout << "\rFrame " << frame_cnt
                          << " rows: " << rows_received << "/" << HEIGHT
                          << " Cap FPS: " << fps << std::flush;
                std::fill(received.begin(), received.end(), 0);
                rows_received = 0;
                last_time = now;
            }
        }
    }

    close(sockfd);
    return 0;
}

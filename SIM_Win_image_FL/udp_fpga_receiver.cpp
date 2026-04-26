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
#include <queue>
#include <sys/socket.h>
#include <opencv2/opencv.hpp>

#define PORT           1234
#define WIDTH          640
#define HEIGHT         480
#define TOTAL_PACKETS  481
#define BATCH          128
#define QUEUE_SIZE     1024

#ifndef SO_RCVBUFFORCE
#define SO_RCVBUFFORCE 33
#endif

// 数据包结构
struct Packet {
    uint32_t pkt_num;
    std::vector<uint8_t> data;
};

// 线程安全队列
class ThreadSafeQueue {
public:
    void push(Packet&& packet) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (queue_.size() < QUEUE_SIZE) {
            queue_.push(std::move(packet));
            cond_.notify_one();
        }
    }

    bool pop(Packet& packet) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (queue_.empty()) {
            return false;
        }
        packet = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void wait_and_pop(Packet& packet) {
        std::unique_lock<std::mutex> lk(mtx_);
        cond_.wait(lk, [this] { return !queue_.empty(); });
        packet = std::move(queue_.front());
        queue_.pop();
    }

private:
    std::queue<Packet> queue_;
    std::mutex mtx_;
    std::condition_variable cond_;
};

// 双缓冲：处理线程写 back，显示线程读 front
static cv::Mat g_front(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0));
static cv::Mat g_back(HEIGHT, WIDTH,  CV_8UC3, cv::Scalar(0));
static std::mutex g_swap_mtx;
static std::atomic<bool> g_new_frame{false};

// ROI（处理线程写，显示线程读，精度要求不高，atomic 足够）
static std::atomic<int> g_x1{0}, g_y1{0}, g_x2{WIDTH}, g_y2{HEIGHT};

// 线程安全队列
static ThreadSafeQueue g_packet_queue;

// FPS 统计
static std::atomic<int>    g_frame_cnt{0};
static std::atomic<double> g_fps{0.0};

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

// ---- 处理线程 ----
void process_thread()
{
    std::vector<uint8_t> received(HEIGHT + 1, 0);
    int rows_received = 0;

    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_cnt  = 0;

    while (true)
    {
        Packet packet;
        g_packet_queue.wait_and_pop(packet);

        uint32_t pkt = packet.pkt_num;
        const auto& data = packet.data;

        // ROI 包
        if (pkt == TOTAL_PACKETS) {
            if (data.size() >= 8) {
                g_x1 = ntohs(*(uint16_t*)(data.data() + 0));
                g_y1 = ntohs(*(uint16_t*)(data.data() + 2));
                g_x2 = ntohs(*(uint16_t*)(data.data() + 4));
                g_y2 = ntohs(*(uint16_t*)(data.data() + 6));
            }
            continue;
        }

        int row = pkt - 1;
        if (row < 0 || row >= HEIGHT) continue;

        uint16_t* pixels;
        if (pkt == 1) {
            if (data.size() < 8 + WIDTH * 2) continue;
            pixels = (uint16_t*)(data.data() + 8);
        } else {
            if (data.size() < WIDTH * 2) continue;
            pixels = (uint16_t*)data.data();
        }

        if (!received[pkt]) {
            received[pkt] = 1;
            rows_received++;
        }

        cv::Vec3b* row_ptr = g_back.ptr<cv::Vec3b>(row);
        for (int i = 0; i < WIDTH; i++) {
            uint16_t p = ntohs(pixels[i]);
            row_ptr[i][0] = (p & 0x1F) << 3;
            row_ptr[i][1] = ((p >> 5) & 0x3F) << 2;
            row_ptr[i][2] = ((p >> 11) & 0x1F) << 3;
        }

        // 收到所有行 → 交换缓冲，更新统计
        if (rows_received == HEIGHT)
        {
            {
                std::lock_guard<std::mutex> lk(g_swap_mtx);
                std::swap(g_front, g_back);
                g_new_frame.store(true, std::memory_order_relaxed);
            }

            frame_cnt++;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - last_time).count();
            double fps = 1.0 / elapsed;

            std::cout << "\rFrame " << frame_cnt
                      << " rows: " << rows_received << "/" << HEIGHT
                      << " FPS: " << fps << std::flush;

            std::fill(received.begin(), received.end(), 0);
            rows_received = 0;
            last_time = now;
        }
    }
}

// ---- 接收线程 ----
void receive_thread(int sockfd)
{
    // 批量接收缓冲
    std::vector<std::vector<uint8_t>> buffers(BATCH, std::vector<uint8_t>(2048));
    std::vector<iovec>   iov(BATCH);
    std::vector<mmsghdr> msgs(BATCH);

    for (int i = 0; i < BATCH; i++) {
        iov[i].iov_base = buffers[i].data();
        iov[i].iov_len  = buffers[i].size();
        msgs[i].msg_hdr.msg_iov    = &iov[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
    }

    long long pkt_count = 0;
    auto stat_time = std::chrono::steady_clock::now();

    while (true)
    {
        int ret = recvmmsg(sockfd, msgs.data(), BATCH, MSG_WAITFORONE, NULL);
        if (ret <= 0) continue;

        pkt_count += ret;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - stat_time).count();
        if (elapsed >= 2.0) {
            fprintf(stderr, "\n[recv] %.0f pkts/s\n", pkt_count / elapsed);
            pkt_count = 0;
            stat_time = now;
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

            // 创建数据包并放入队列
            Packet packet;
            packet.pkt_num = pkt;
            packet.data.resize(payload_len);
            std::memcpy(packet.data.data(), payload, payload_len);
            g_packet_queue.push(std::move(packet));
        }
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
    bind(sockfd, (struct sockaddr *)&addr, sizeof(addr));

    std::cout << "UDP接收启动: port=" << PORT
              << "  RCVBUFF=" << actual / 1024 << " KB" << std::endl;

    // 启动显示线程
    std::thread disp(display_thread);
    disp.detach();

    // 启动处理线程
    std::thread proc(process_thread);
    proc.detach();

    // 启动接收线程
    std::thread recv(receive_thread, sockfd);
    recv.join();

    close(sockfd);
    return 0;
}

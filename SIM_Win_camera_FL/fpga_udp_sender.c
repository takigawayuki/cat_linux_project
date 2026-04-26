#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
  #pragma comment(lib, "ws2_32.lib")
  typedef SOCKET sock_t;
  #define CLOSE_SOCK(s) closesocket(s)

  static inline long long get_time_ns() {
      LARGE_INTEGER freq, cnt;
      QueryPerformanceFrequency(&freq);
      QueryPerformanceCounter(&cnt);
      return (cnt.QuadPart * 1000000000LL) / freq.QuadPart;
  }
  static inline void sleep_ns(long long ns) {
      Sleep((DWORD)(ns / 1000000));
  }
#else
  #include <sys/socket.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <time.h>
  typedef int sock_t;
  #define CLOSE_SOCK(s) close(s)

  static inline long long get_time_ns() {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return ts.tv_sec * 1000000000LL + ts.tv_nsec;
  }
  static inline void sleep_ns(long long ns) {
      struct timespec ts = { ns / 1000000000LL, ns % 1000000000LL };
      nanosleep(&ts, NULL);
  }
#endif

#include <opencv2/opencv.hpp>

#define TARGET_IP    "192.168.137.5"
#define TARGET_PORT  1234
#define IMG_WIDTH    640
#define IMG_HEIGHT   480
#define FRAME_HEAD   0xF05AA50F
#define FPS          60

#define ROI_X1  160
#define ROI_Y1  120
#define ROI_X2  480
#define ROI_Y2  360

// 包1: 4(pkt_num) + 4(frame_head) + 4(WxH) + WIDTH*2(pixels)
#define PKT1_SIZE   (4 + 4 + 4 + IMG_WIDTH * 2)
// 包2-480: 4(pkt_num) + WIDTH*2(pixels)
#define PKTN_SIZE   (4 + IMG_WIDTH * 2)
// 包481: 4(pkt_num) + 8(roi)
#define PKT481_SIZE (4 + 8)

static uint8_t pkt1_buf[PKT1_SIZE];
static uint8_t pktn_buf[PKTN_SIZE];
static uint8_t pkt481_buf[PKT481_SIZE];

static inline uint16_t bgr_to_rgb565(uint8_t b, uint8_t g, uint8_t r)
{
    return (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

static void send_frame(sock_t sock, struct sockaddr_in *dst, cv::Mat &frame)
{
    // --- 包1 ---
    uint32_t pkt_num = htonl(1);
    uint32_t fhead   = htonl(FRAME_HEAD);
    uint16_t w       = htons(IMG_WIDTH);
    uint16_t h       = htons(IMG_HEIGHT);
    memcpy(pkt1_buf + 0, &pkt_num, 4);
    memcpy(pkt1_buf + 4, &fhead,   4);
    memcpy(pkt1_buf + 8, &w,       2);
    memcpy(pkt1_buf + 10, &h,      2);

    uint8_t *row0 = frame.ptr<uint8_t>(0);
    uint16_t *dst_px = (uint16_t *)(pkt1_buf + 12);
    for (int i = 0; i < IMG_WIDTH; i++) {
        uint16_t px = bgr_to_rgb565(row0[i*3], row0[i*3+1], row0[i*3+2]);
        dst_px[i] = htons(px);
    }
    sendto(sock, (char*)pkt1_buf, PKT1_SIZE, 0,
           (struct sockaddr*)dst, sizeof(*dst));

    // --- 包2-480 ---
    for (int row = 1; row < IMG_HEIGHT; row++) {
        uint32_t n = htonl((uint32_t)(row + 1));
        memcpy(pktn_buf, &n, 4);

        uint8_t *rowp = frame.ptr<uint8_t>(row);
        uint16_t *dp  = (uint16_t *)(pktn_buf + 4);
        for (int i = 0; i < IMG_WIDTH; i++) {
            uint16_t px = bgr_to_rgb565(rowp[i*3], rowp[i*3+1], rowp[i*3+2]);
            dp[i] = htons(px);
        }
        sendto(sock, (char*)pktn_buf, PKTN_SIZE, 0,
               (struct sockaddr*)dst, sizeof(*dst));
    }

    // --- 包481 (ROI) ---
    uint32_t n481 = htonl(481);
    memcpy(pkt481_buf, &n481, 4);
    uint16_t roi[4] = { htons(ROI_X1), htons(ROI_Y1), htons(ROI_X2), htons(ROI_Y2) };
    memcpy(pkt481_buf + 4, roi, 8);
    sendto(sock, (char*)pkt481_buf, PKT481_SIZE, 0,
           (struct sockaddr*)dst, sizeof(*dst));
}

int main(void)
{
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);
#endif

    sock_t sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }

    // 增大发送缓冲
    int sndbuf = 4 * 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char*)&sndbuf, sizeof(sndbuf));

    struct sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(TARGET_PORT);
    inet_pton(AF_INET, TARGET_IP, &dst.sin_addr);

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  IMG_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT);
    if (!cap.isOpened()) { fprintf(stderr, "Cannot open camera\n"); return 1; }

    const long long frame_ns = 1000000000LL / FPS;
    int frame_count = 0;

    printf("Sending to %s:%d  target=%d fps\n", TARGET_IP, TARGET_PORT, FPS);

    while (1) {
        long long t0 = get_time_ns();

        cv::Mat frame;
        if (!cap.read(frame)) { fprintf(stderr, "capture failed\n"); break; }

        send_frame(sock, &dst, frame);
        frame_count++;
        printf("\rFrame %d", frame_count);
        fflush(stdout);

        long long t1 = get_time_ns();
        long long elapsed = t1 - t0;
        long long wait = frame_ns - elapsed;
        if (wait > 0) sleep_ns(wait);
    }

    cap.release();
    CLOSE_SOCK(sock);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

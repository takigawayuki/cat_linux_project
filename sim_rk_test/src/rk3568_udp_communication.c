/**
 * RK3568与FPGA UDP通信程序 (C语言版本)
 * 
 * 编译命令:
 *   gcc rk3568_udp_communication.c -o rk3568_udp
 * 
 * 运行:
 *   ./rk3568_udp
 * 
 * FPGA端配置:
 *   - FPGA IP: 192.168.1.110
 *   - FPGA端口: 8080
 *   - RK3568 IP: 192.168.1.105
 *   - RK3568端口: 8080
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>

#define BUFFER_SIZE 1024
#define LOCAL_IP "192.168.1.105"
#define LOCAL_PORT 8080
#define FPGA_IP "192.168.1.110"
#define FPGA_PORT 8080

// 全局变量
static int running = 1;
static int sock_fd = -1;

/**
 * 获取当前时间戳字符串
 */
void get_timestamp(char *buffer, size_t size) {
    struct timespec ts;
    struct tm *tm_info;
    
    clock_gettime(CLOCK_REALTIME, &ts);
    tm_info = localtime(&ts.tv_sec);
    
    snprintf(buffer, size, "%04d-%02d-%02d %02d:%02d:%02d.%03ld",
             tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday,
             tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec,
             ts.tv_nsec / 1000000);
}

/**
 * 信号处理函数
 */
void signal_handler(int sig) {
    printf("\n[INFO] 收到信号 %d，正在退出...\n", sig);
    running = 0;
}

/**
 * 打印数据内容（多种格式）
 */
void print_data(const char *prefix, const unsigned char *data, int len) {
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("[%s] %s 数据长度: %d 字节\n", timestamp, prefix, len);
    printf("[%s] %s 数据内容 (hex): ", timestamp, prefix);
    for (int i = 0; i < len && i < 32; i++) {
        printf("%02X ", data[i]);
    }
    if (len > 32) printf("...");
    printf("\n");
    
    printf("[%s] %s 数据内容 (ascii): ", timestamp, prefix);
    for (int i = 0; i < len && i < 64; i++) {
        if (data[i] >= 32 && data[i] <= 126) {
            printf("%c", data[i]);
        } else {
            printf(".");
        }
    }
    if (len > 64) printf("...");
    printf("\n");
}

/**
 * 接收线程函数
 */
void *receive_thread_func(void *arg) {
    unsigned char buffer[BUFFER_SIZE];
    struct sockaddr_in from_addr;
    socklen_t from_len = sizeof(from_addr);
    
    printf("[INFO] 接收线程已启动\n");
    
    while (running) {
        // 设置超时
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        
        if (setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
            perror("[ERROR] 设置接收超时失败");
            break;
        }
        
        // 接收数据
        int recv_len = recvfrom(sock_fd, buffer, BUFFER_SIZE - 1, 0,
                               (struct sockaddr *)&from_addr, &from_len);
        
        if (recv_len < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 超时是正常的，继续循环
                continue;
            }
            perror("[ERROR] 接收数据失败");
            break;
        }
        
        // 打印接收到的数据
        char timestamp[32];
        get_timestamp(timestamp, sizeof(timestamp));
        printf("\n[%s] 收到来自 %s:%d 的数据\n", timestamp,
               inet_ntoa(from_addr.sin_addr), ntohs(from_addr.sin_port));
        
        print_data("RECV", buffer, recv_len);
        printf("\n");
    }
    
    printf("[INFO] 接收线程已退出\n");
    return NULL;
}

/**
 * 发送数据到FPGA
 */
int send_to_fpga(const char *data, int len) {
    struct sockaddr_in fpga_addr;
    
    memset(&fpga_addr, 0, sizeof(fpga_addr));
    fpga_addr.sin_family = AF_INET;
    fpga_addr.sin_port = htons(FPGA_PORT);
    
    if (inet_pton(AF_INET, FPGA_IP, &fpga_addr.sin_addr) <= 0) {
        perror("[ERROR] 无效的FPGA IP地址");
        return -1;
    }
    
    int sent_len = sendto(sock_fd, data, len, 0,
                         (struct sockaddr *)&fpga_addr, sizeof(fpga_addr));
    
    if (sent_len < 0) {
        perror("[ERROR] 发送数据失败");
        return -1;
    }
    
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    printf("[%s] 发送数据到 %s:%d\n", timestamp, FPGA_IP, FPGA_PORT);
    print_data("SEND", (unsigned char *)data, sent_len);
    
    return sent_len;
}

/**
 * 主函数
 */
int main(int argc, char *argv[]) {
    int ret;
    pthread_t recv_thread;
    struct sockaddr_in local_addr;
    
    printf("========================================\n");
    printf("RK3568与FPGA UDP通信测试程序 (C版本)\n");
    printf("========================================\n");
    
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 创建UDP socket
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0) {
        perror("[ERROR] 创建socket失败");
        return -1;
    }
    
    // 设置地址重用
    int reuse = 1;
    if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("[ERROR] 设置地址重用失败");
        close(sock_fd);
        return -1;
    }
    
    // 绑定本地地址
    memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(LOCAL_PORT);
    local_addr.sin_addr.s_addr = inet_addr(LOCAL_IP);
    
    if (bind(sock_fd, (struct sockaddr *)&local_addr, sizeof(local_addr)) < 0) {
        perror("[ERROR] 绑定地址失败");
        close(sock_fd);
        return -1;
    }
    
    printf("[INFO] UDP Socket初始化成功\n");
    printf("[INFO] 本机地址: %s:%d\n", LOCAL_IP, LOCAL_PORT);
    printf("[INFO] FPGA地址: %s:%d\n", FPGA_IP, FPGA_PORT);
    
    // 启动接收线程
    ret = pthread_create(&recv_thread, NULL, receive_thread_func, NULL);
    if (ret != 0) {
        printf("[ERROR] 创建接收线程失败: %s\n", strerror(ret));
        close(sock_fd);
        return -1;
    }
    
    printf("\n[提示] 等待FPGA发送数据...\n");
    printf("[提示] 输入 'help' 查看命令列表\n");
    printf("----------------------------------------\n");
    
    // 主循环 - 处理用户输入
    char input[BUFFER_SIZE];
    while (running) {
        printf("\n请输入要发送的数据 (或输入 'quit' 退出): ");
        fflush(stdout);
        
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }
        
        // 去除换行符
        input[strcspn(input, "\n")] = 0;
        
        if (strlen(input) == 0) {
            continue;
        }
        
        if (strcmp(input, "quit") == 0) {
            printf("[INFO] 正在退出程序...\n");
            break;
        } else if (strcmp(input, "help") == 0) {
            printf("\n命令列表:\n");
            printf("  help     - 显示帮助信息\n");
            printf("  quit     - 退出程序\n");
            printf("  test     - 发送测试数据\n");
            printf("  <数据>   - 发送任意数据到FPGA\n");
            printf("\n当前状态:\n");
            printf("  本机: %s:%d\n", LOCAL_IP, LOCAL_PORT);
            printf("  FPGA: %s:%d\n", FPGA_IP, FPGA_PORT);
        } else if (strcmp(input, "test") == 0) {
            // 发送测试数据
            const char *test_data = "RK3568_TEST_DATA";
            send_to_fpga(test_data, strlen(test_data));
        } else {
            // 发送用户输入的数据
            send_to_fpga(input, strlen(input));
        }
    }
    
    // 清理资源
    running = 0;
    pthread_join(recv_thread, NULL);
    close(sock_fd);
    
    printf("[INFO] UDP通信已停止\n");
    return 0;
}
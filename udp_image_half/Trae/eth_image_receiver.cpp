#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>

#define UDP_PORT 1234
#define PACKET_COUNT 961
#define IMAGE_PACKETS 960
#define COORDINATE_PACKET 960 // 0-based index
#define MAX_PACKET_SIZE 1500

using namespace cv;
using namespace std;

struct Coordinates {
    uint16_t x1, y1, x2, y2;
};

int main() {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);
    char buffer[MAX_PACKET_SIZE];
    uint8_t image_buffer[640 * 480 * 2]; // RGB565
    uint8_t rgb888_buffer[640 * 480 * 3]; // RGB888
    Coordinates coords = {0, 0, 639, 479}; // Default to full image
    int current_pixel = 0;
    
    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(UDP_PORT);
    
    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    cout << "Listening for UDP packets on port " << UDP_PORT << "..." << endl;
    
    while (true) {
        ssize_t bytes_received = recvfrom(sockfd, buffer, MAX_PACKET_SIZE, 0, (struct sockaddr *)&client_addr, &addr_len);
        if (bytes_received < 0) {
            perror("recvfrom failed");
            continue;
        }
        
        // Extract packet number (4 bytes, big-endian)
        int current_packet = ntohl(*(uint32_t *)buffer);
        
        if (current_packet == 0) {
            // First packet - contains header, resolution, and image data
            // Skip 4-byte packet number + 4-byte frame header + 4-byte resolution = 12 bytes
            int data_start = 12;
            int data_size = bytes_received - data_start;
            memcpy(image_buffer + current_pixel, buffer + data_start, data_size);
            current_pixel += data_size;
        } else if (current_packet < IMAGE_PACKETS) {
            // Middle image packets - contains packet number + image data
            // Skip 4-byte packet number
            int data_start = 4;
            int data_size = bytes_received - data_start;
            memcpy(image_buffer + current_pixel, buffer + data_start, data_size);
            current_pixel += data_size;
        } else if (current_packet == IMAGE_PACKETS) {
            // Last packet - contains coordinates
            // Skip 4-byte packet number, then read 2-byte coordinates (big-endian)
            coords.x1 = ntohs(*(uint16_t *)(buffer + 4));
            coords.y1 = ntohs(*(uint16_t *)(buffer + 6));
            coords.x2 = ntohs(*(uint16_t *)(buffer + 8));
            coords.y2 = ntohs(*(uint16_t *)(buffer + 10));
            
            // Ensure coordinates are within bounds
            coords.x1 = max(0, min(639, (int)coords.x1));
            coords.y1 = max(0, min(479, (int)coords.y1));
            coords.x2 = max(0, min(639, (int)coords.x2));
            coords.y2 = max(0, min(479, (int)coords.y2));
            
            // Swap coordinates if needed to ensure x1 <= x2 and y1 <= y2
            if (coords.x1 > coords.x2) swap(coords.x1, coords.x2);
            if (coords.y1 > coords.y2) swap(coords.y1, coords.y2);
            
            // Convert RGB565 to RGB888
            for (int i = 0; i < 640 * 480; i++) {
                uint16_t rgb565 = *(uint16_t *)(image_buffer + i * 2);
                uint8_t r = ((rgb565 >> 11) & 0x1F) << 3;
                uint8_t g = ((rgb565 >> 5) & 0x3F) << 2;
                uint8_t b = (rgb565 & 0x1F) << 3;
                rgb888_buffer[i * 3] = r;
                rgb888_buffer[i * 3 + 1] = g;
                rgb888_buffer[i * 3 + 2] = b;
            }
            
            // Create OpenCV Mat for full image
            Mat full_image(480, 640, CV_8UC3, rgb888_buffer);
            
            // Crop to the specified coordinates
            Mat cropped_image = full_image(Rect(coords.x1, coords.y1, coords.x2 - coords.x1 + 1, coords.y2 - coords.y1 + 1));
            
            // Display both images
            imshow("Full Image", full_image);
            imshow("Cropped Image", cropped_image);
            
            // Reset for next frame
            current_pixel = 0;
            
            // Check for exit key
            if (waitKey(1) == 27) break;
        }
    }
    
    close(sockfd);
    destroyAllWindows();
    return 0;
}

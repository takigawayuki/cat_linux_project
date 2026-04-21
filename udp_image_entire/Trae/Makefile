# Makefile for eth_receiver

CC = g++
CFLAGS = -Wall -O2 -I/usr/include/opencv4
LDFLAGS = -L/usr/lib/aarch64-linux-gnu -lopencv_core -lopencv_highgui -lopencv_imgproc -lpthread

TARGET = eth_receiver
SRC = eth_receiver.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
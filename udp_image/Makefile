CXX = g++
CXXFLAGS = -O3 -std=c++11

TARGET = udp_receiver
SRC = udp_fpga_receiver.cpp

OPENCV = `pkg-config --cflags --libs opencv4`

all:
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS) $(OPENCV)

clean:
	rm -f $(TARGET)
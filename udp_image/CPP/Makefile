CXX = g++
CXXFLAGS = -O3 -march=native -std=c++11

TARGET = udp_receiver
SRC = udp_fpga_receiver.cpp

all:
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS) `pkg-config --cflags --libs opencv4`

clean:
	rm -f $(TARGET)
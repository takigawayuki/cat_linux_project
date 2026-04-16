CXX = g++
CXXFLAGS = -O3 -std=c++11
TARGET = udp_receiver
SRC = udp_fpga_receiver.cpp

OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

all:
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS) $(OPENCV_FLAGS)

run:
	./$(TARGET)

clean:
	rm -f $(TARGET)
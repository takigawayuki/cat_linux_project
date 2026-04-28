CXX = g++
CXXFLAGS = -O3 -std=c++11

TARGET = udp_320x240_receiver
SRC = udp_320x240_receiver.cpp

OPENCV = `pkg-config --cflags --libs opencv4`

all:
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS) $(OPENCV) -lpthread

clean:
	rm -f $(TARGET)
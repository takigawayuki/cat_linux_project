CC = g++
TARGET = udp_receiver

SRCS = udp_receiver.cpp
OBJS = $(SRCS:.cpp=.o)

CFLAGS = -O2 -Wall -std=c++11
LDFLAGS = -lpthread

# ===== 自动检测 OpenCV =====
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null)

ifeq ($(OPENCV_LIBS),)
    OPENCV_CFLAGS := $(shell pkg-config --cflags opencv 2>/dev/null)
    OPENCV_LIBS   := $(shell pkg-config --libs opencv 2>/dev/null)
endif

ifeq ($(OPENCV_LIBS),)
    $(error ❌ OpenCV not found. Please install opencv4 or set pkg-config path)
endif

CFLAGS += $(OPENCV_CFLAGS)
LDFLAGS += $(OPENCV_LIBS)

# ===== 编译规则 =====
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

info:
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
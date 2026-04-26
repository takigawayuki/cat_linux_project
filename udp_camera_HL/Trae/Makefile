CC = g++
CFLAGS = -Wall -Wextra -O2
LDFLAGS = `pkg-config --cflags --libs opencv4`

TARGET = eth_image_receiver
SRC = eth_image_receiver.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

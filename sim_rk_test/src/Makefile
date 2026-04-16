# RK3568 UDP通信程序 Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O2 -pthread
TARGET = rk3568_udp
SOURCE = rk3568_udp_communication.c

# 默认目标
all: $(TARGET)

# 编译目标
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)
	@echo "编译成功: $(TARGET)"

# 清理
clean:
	rm -f $(TARGET)
	@echo "清理完成"

# 安装到系统路径
install: $(TARGET)
	@echo "请使用 sudo cp $(TARGET) /usr/local/bin/ 进行安装"

# 运行
run: $(TARGET)
	./$(TARGET)

# 帮助
help:
	@echo "可用命令:"
	@echo "  make      - 编译程序"
	@echo "  make clean - 清理编译文件"
	@echo "  make run   - 编译并运行程序"
	@echo "  make help  - 显示此帮助信息"

.PHONY: all clean install run help
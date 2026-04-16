# RK3568与FPGA UDP通信使用说明

## 一、网络配置

### FPGA端配置
- **IP地址**: 192.168.1.110
- **端口**: 8080
- **MAC地址**: 11:11:11:11:11:11
- **目标IP**: 192.168.1.105 (RK3568)
- **目标端口**: 8080

### RK3568端配置
- **IP地址**: 192.168.1.105
- **端口**: 8080
- **目标IP**: 192.168.1.110 (FPGA)
- **目标端口**: 8080

## 二、网络连接设置

### 1. 配置RK3568网络接口

```bash
# 查看网络接口
ip addr show

# 配置静态IP（假设使用eth0接口）
sudo ip addr add 192.168.1.105/24 dev eth0
sudo ip link set eth0 up

# 或者编辑网络配置文件（永久配置）
sudo nano /etc/network/interfaces
```

添加以下内容：
```
auto eth0
iface eth0 inet static
    address 192.168.1.105
    netmask 255.255.255.0
    gateway 192.168.1.1
```

### 2. 测试网络连通性

```bash
# 测试与FPGA的连通性
ping 192.168.1.110

# 测试UDP端口连通性
nc -u -zv 192.168.1.110 8080
```

## 三、使用Python版本（推荐用于快速测试）

### 1. 安装依赖
```bash
# RK3568通常已预装Python3
python3 --version

# 如需安装额外依赖
pip3 install --upgrade pip
```

### 2. 运行程序
```bash
# 赋予执行权限
chmod +x rk3568_udp_communication.py

# 运行程序
python3 rk3568_udp_communication.py
```

### 3. 交互式命令
- `help` - 显示帮助信息
- `quit` - 退出程序
- `test` - 发送测试数据
- `<任意数据>` - 发送自定义数据到FPGA

### 4. 代码示例

```python
# 在其他Python程序中使用
from rk3568_udp_communication import RK3568_UDP_Communication

# 创建通信对象
udp = RK3568_UDP_Communication()

# 启动接收线程
udp.start_receive_thread()

# 发送数据
udp.send_data("Hello FPGA!")

# 发送命令
udp.send_command("CMD_START")

# 停止通信
udp.stop()
```

## 四、使用C语言版本（推荐用于高性能应用）

### 1. 编译程序
```bash
# 使用Makefile编译
make

# 或手动编译
gcc -Wall -Wextra -O2 -pthread -o rk3568_udp rk3568_udp_communication.c
```

### 2. 运行程序
```bash
# 运行
./rk3568_udp

# 或使用make run
make run
```

### 3. 交叉编译（在PC上为RK3568编译）

如果需要在PC上交叉编译，需要安装交叉编译工具链：

```bash
# 安装交叉编译工具链（Ubuntu/Debian）
sudo apt-get install gcc-aarch64-linux-gnu

# 交叉编译
aarch64-linux-gnu-gcc -Wall -Wextra -O2 -pthread -o rk3568_udp rk3568_udp_communication.c

# 传输到RK3568
scp rk3568_udp root@192.168.1.105:/root/
```

## 五、FPGA发送的数据格式

根据FPGA代码，FPGA会每秒发送一次20字节的测试数据：

```c
// FPGA发送的测试数据
// "www.meyesemi.com  \n"
// 十六进制: 77 77 77 2E 6D 65 79 65 73 65 6D 69 2E 63 6F 6D 20 20 20 0A
```

### 数据解析示例

```python
# Python解析示例
data = b'www.meyesemi.com  \n'
print(f"原始数据: {data}")
print(f"十六进制: {data.hex()}")
print(f"ASCII: {data.decode('ascii')}")
```

```c
// C语言解析示例
unsigned char data[20] = {0x77, 0x77, 0x77, 0x2E, 0x6D, 0x65, 0x79, 0x65, 
                          0x73, 0x65, 0x6D, 0x69, 0x2E, 0x63, 0x6F, 0x6D, 
                          0x20, 0x20, 0x20, 0x0A};

printf("原始数据: ");
for (int i = 0; i < 20; i++) {
    printf("%02X ", data[i]);
}
printf("\n");

printf("ASCII: %.20s\n", data);
```

## 六、常见问题排查

### 1. 无法接收FPGA数据

**检查项：**
- 确认RK3568和FPGA在同一网段
- 检查防火墙设置
- 确认IP地址配置正确
- 使用Wireshark抓包分析

```bash
# 检查防火墙
sudo iptables -L -n

# 临时关闭防火墙（测试用）
sudo iptables -F

# 使用tcpdump抓包
sudo tcpdump -i eth0 -n udp port 8080
```

### 2. 发送数据失败

**检查项：**
- 确认FPGA的IP地址正确
- 检查网络连通性
- 确认FPGA端已启动并正常工作

### 3. 性能优化

**Python版本优化：**
- 使用多线程处理大量数据
- 增加接收缓冲区大小

```python
# 增加缓冲区大小
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)
```

**C语言版本优化：**
- 使用零拷贝技术
- 批量处理数据
- 使用DMA加速

## 七、高级应用示例

### 1. 双向通信示例

```python
# RK3568端
udp = RK3568_UDP_Communication()
udp.start_receive_thread()

# 发送控制命令
udp.send_command("CMD_START")
time.sleep(1)
udp.send_command("CMD_STOP")
```

### 2. 数据采集示例

```python
# 持续采集FPGA数据
import csv

udp = RK3568_UDP_Communication()
udp.start_receive_thread()

with open('fpga_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'data_hex', 'data_ascii'])
    
    for _ in range(100):  # 采集100次
        time.sleep(1)
        # 接收数据会自动打印，可以修改代码保存到文件
```

### 3. 实时监控示例

```c
// C语言实时监控
while (running) {
    // 接收线程自动运行
    // 主线程可以处理其他任务
    process_data();
    control_fpga();
    sleep(0.1);
}
```

## 八、安全建议

1. **网络安全**
   - 在生产环境中使用加密通信
   - 实现数据校验机制
   - 添加访问控制

2. **错误处理**
   - 实现完善的错误处理机制
   - 添加重连逻辑
   - 记录日志

3. **性能监控**
   - 监控网络延迟
   - 统计丢包率
   - 监控CPU和内存使用

## 九、技术支持

如有问题，请检查：
1. FPGA和RK3568的网络配置
2. 防火墙设置
3. 程序日志输出
4. 网络抓包分析

## 十、版本历史

- v1.0 - 初始版本
  - Python版本
  - C语言版本
  - 基础文档
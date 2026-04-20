#!/bin/bash

# 检查OpenCV安装情况
echo "=== OpenCV Installation Check ==="

# 检查pkg-config是否存在
if command -v pkg-config &> /dev/null; then
    echo "pkg-config found"
else
    echo "pkg-config not found"
fi

# 检查OpenCV 4
if pkg-config --exists opencv4; then
    echo "OpenCV 4 found"
    echo "CFLAGS: $(pkg-config --cflags opencv4)"
    echo "LDFLAGS: $(pkg-config --libs opencv4)"
elif pkg-config --exists opencv; then
    echo "OpenCV (older version) found"
    echo "CFLAGS: $(pkg-config --cflags opencv)"
    echo "LDFLAGS: $(pkg-config --libs opencv)"
else
    echo "OpenCV not found via pkg-config"
    
    # 检查常见安装路径
    echo "Checking common OpenCV installation paths..."
    
    # 检查头文件
    if [ -f "/usr/include/opencv4/opencv2/opencv.hpp" ]; then
        echo "OpenCV headers found at /usr/include/opencv4"
    elif [ -f "/usr/include/opencv2/opencv.hpp" ]; then
        echo "OpenCV headers found at /usr/include"
    else
        echo "OpenCV headers not found in common locations"
    fi
    
    # 检查库文件
    if ls /usr/lib/*opencv*.so &> /dev/null; then
        echo "OpenCV libraries found in /usr/lib"
        ls /usr/lib/*opencv*.so
    elif ls /usr/lib64/*opencv*.so &> /dev/null; then
        echo "OpenCV libraries found in /usr/lib64"
        ls /usr/lib64/*opencv*.so
    else
        echo "OpenCV libraries not found in common locations"
    fi
fi

echo "=== Check Complete ==="

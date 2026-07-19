# RKNN 车牌串联推理脚本使用说明

本文说明 `fenqusai/tools/pipeline/infer_rknn_plate.py` 的用途、运行方式、参数含义和常见排查方法。

这个脚本用于在瑞芯微板端验证当前项目的 **YOLO 检测 + LPRNet 识别** 串联推理：

```text
输入图片 / 图片文件夹
  -> YOLO RKNN 检测 plate/person/car/traffic_light
  -> CPU 做 YOLO 9 输出后处理
  -> 裁剪 plate 框
  -> LPRNet RKNN 输出 logits=(1,74,18)
  -> CPU 做 constrained CTC beam search
  -> 输出检测框、车牌类型、车牌文本和可视化图片
```

## 1. 脚本位置

```bash
fenqusai/tools/pipeline/infer_rknn_plate.py
```

建议从项目根目录运行：

```bash
cd /home/cat/cat_linux_project
```

## 2. 依赖环境

脚本需要在 RKNN 板端环境运行，至少需要：

```text
python3
opencv-python / cv2
numpy
rknn-toolkit-lite2 / rknnlite.api.RKNNLite
```

如果只执行 `--help`，脚本可以在缺少 `cv2/numpy/rknnlite` 的环境里显示参数；但真正推理必须在板端或已安装 RKNNLite 的环境执行。

## 3. 默认使用的模型

脚本默认模型路径：

```text
YOLO:
  fenqusai/rknn/best.rknn

LPRNet:
  fenqusai/rknn/lprnet_unified_p15_focus_fp.rknn
```

其中：

```text
best.rknn:
  4 类 YOLO 检测模型
  classes = plate / person / car / traffic_light
  预期输出为 9 outputs

lprnet_unified_p15_focus_fp.rknn:
  当前建议优先验证的 LPRNet 主模型
  预期输出 logits = (1, 74, 18)
```

如果要切换 LPRNet INT8 或 p2 备选模型，用 `--lpr-model` 指定即可。

## 4. 快速运行

### 4.1 跑单张图片

```bash
python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image test_photo/Random/03875-89_268-218\&516_566\&618-561\&606_222\&618_218\&525_566\&516-0_0_3_25_30_29_29_32-97-148.jpg \
  --output-dir fenqusai/tools/pipeline/result_rknn_plate \
  --core-mask default \
  --lpr-core-mask default
```

如果路径里没有 shell 特殊字符，也可以直接写普通路径。含 `&` 的 CCPD 文件名建议加引号：

```bash
python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image 'test_photo/Random/03875-89_268-218&516_566&618-561&606_222&618_218&525_566&516-0_0_3_25_30_29_29_32-97-148.jpg' \
  --output-dir fenqusai/tools/pipeline/result_rknn_plate \
  --core-mask default \
  --lpr-core-mask default
```

### 4.2 跑图片文件夹

```bash
python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image test_photo/Random \
  --output-dir fenqusai/tools/pipeline/result_rknn_plate \
  --core-mask default \
  --lpr-core-mask default
```

脚本会遍历目录下的：

```text
.jpg / .jpeg / .png / .bmp
```

### 4.3 指定模型

```bash
python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image test_photo/Random \
  --yolo-model fenqusai/rknn/best.rknn \
  --lpr-model fenqusai/rknn/lprnet_unified_p15_focus_fp.rknn \
  --output-dir fenqusai/tools/pipeline/result_rknn_plate \
  --core-mask default \
  --lpr-core-mask default
```

切换 LPRNet INT8：

```bash
python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image test_photo/Random \
  --lpr-model fenqusai/rknn/lprnet_unified_p15_focus_i8.rknn \
  --output-dir fenqusai/tools/pipeline/result_rknn_plate_i8 \
  --core-mask default \
  --lpr-core-mask default
```

## 5. 输出结果

脚本每处理一张图，会在终端打印一行 JSON：

```json
{
  "image": "test_photo/Random/example.jpg",
  "detections": [
    {
      "class": "plate",
      "score": 0.91,
      "box": [218, 516, 566, 618]
    }
  ],
  "plates": [
    {
      "box": [218, 516, 566, 618],
      "det_score": 0.91,
      "plate_type": "blue",
      "plate_text": "粤B12345",
      "lpr_score": 0.86,
      "beam_score": -12.34,
      "raw_text": "粤B12345",
      "crop_path": "fenqusai/tools/pipeline/result_rknn_plate/crops/example_plate0.jpg"
    }
  ]
}
```

字段说明：

```text
detections:
  YOLO 检测结果，包含 plate/person/car/traffic_light。

plates:
  只对 class=plate 的框继续做 LPRNet 识别。

det_score:
  YOLO 的 max(cls)，不是 cls_sum。

plate_type:
  先由 HSV 粗判颜色，再由 constrained decode fallback 确认。
  可能值包括 blue / green / yellow / black / special_7 / unknown_8 等。

plate_text:
  constrained CTC beam search 的最终车牌文本。

lpr_score:
  beam score 按时间步归一化后的粗略概率，主要用于排序和调试。

beam_score:
  原始 beam log score，数值通常为负，越大越好。

raw_text:
  greedy CTC 的调试输出，不作为最终结果。

crop_path:
  保存的 plate crop 路径。
```

默认会保存：

```text
可视化图片:
  <output-dir>/<原图名>_vis.jpg

车牌裁剪:
  <output-dir>/crops/<原图名>_plate0.jpg
```

如果不想保存：

```bash
--no-save-vis
--no-save-crops
```

## 6. 关键参数

### 6.1 检测阈值

```bash
--conf-thres 0.25
--nms-thres 0.45
```

建议：

```text
漏检 plate:
  降低 --conf-thres，例如 0.20。

误检太多:
  提高 --conf-thres，例如 0.35。

重复框太多:
  降低 --nms-thres，例如 0.35。
```

### 6.2 YOLO 输入通道

```bash
--yolo-color rgb
--yolo-color bgr
```

默认是：

```text
--yolo-color rgb
```

注意：YOLO 输入 RGB/BGR 必须和训练导出、RKNN 转换配置、demo 输入方式一致。

如果出现下面现象，优先试试切换通道：

```text
整图完全检测不到
检测框非常乱
分数普遍异常低
```

对比命令：

```bash
python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image test_photo/Random \
  --yolo-color rgb \
  --output-dir fenqusai/tools/pipeline/result_rgb \
  --core-mask default \
  --lpr-core-mask default

python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image test_photo/Random \
  --yolo-color bgr \
  --output-dir fenqusai/tools/pipeline/result_bgr \
  --core-mask default \
  --lpr-core-mask default
```

哪一组检测框和 ONNX/PyTorch 更接近，就用哪一组。

### 6.3 LPRNet 输入通道

```bash
--lpr-color bgr
--lpr-color rgb
```

默认是：

```text
--lpr-color bgr
```

当前 LPRNet 转换脚本使用 `mean_values/std_values` 做归一化时，板端只需要输入 `uint8` crop。

也就是说：

```text
如果 RKNN config 已设置:
  mean_values = [127.5, 127.5, 127.5]
  std_values  = [128.0, 128.0, 128.0]

板端输入:
  BGR/RGB uint8 crop，resize 到 94x24

不要再手动做:
  (img - 127.5) / 128
```

否则会重复归一化。

### 6.4 beam search 参数

```bash
--beam-width 10
--beam-topk 8
```

默认值对应文档里建议的 constrained decode 配置。

调参建议：

```text
识别不稳、局部字符容易错:
  可以试 --beam-width 20。

CPU 占用高:
  可以试 --beam-width 5 或 --beam-topk 5。
```

### 6.5 crop padding

```bash
--crop-pad-x 0.08
--crop-pad-y 0.15
```

含义：

```text
水平方向左右各扩 8%
垂直方向上下各扩 15%
```

如果车牌裁剪太紧、首尾字符容易掉，适当加大：

```bash
--crop-pad-x 0.10 --crop-pad-y 0.20
```

如果 crop 包含太多背景，适当减小。

### 6.6 NPU 核心绑定

RK3568 运行时请使用：

```bash
--core-mask default
--lpr-core-mask default
```

RK3568 不支持手动设置 `core_mask`。如果设置 `core0/core012/auto` 这类参数，会报：

```text
The core_mask is only supported by ['RK3588', 'RK3576'].
```

原因是 RKNNLite 的 `core_mask` 只支持 RK3588/RK3576 这类多 NPU 核平台；RK3568 不支持手动绑核。

第一版验证优先保证能跑通：

```bash
--core-mask default --lpr-core-mask default
```

如果换到 RK3588 / RK3576，后面做性能优化时可以尝试：

```bash
--core-mask core012 --lpr-core-mask core0
--core-mask core012 --lpr-core-mask default
--core-mask auto --lpr-core-mask auto
```

LPRNet 很小，不一定需要占满三个 NPU 核。

## 7. 当前后处理逻辑

### 7.1 YOLO 后处理

脚本按 9 输出 RKNN 友好 YOLO 处理：

```text
80x80:
  bbox:    (1,64,80,80)
  cls:     (1,4,80,80)
  cls_sum: (1,1,80,80)

40x40:
  bbox:    (1,64,40,40)
  cls:     (1,4,40,40)
  cls_sum: (1,1,40,40)

20x20:
  bbox:    (1,64,20,20)
  cls:     (1,4,20,20)
  cls_sum: (1,1,20,20)
```

处理顺序：

```text
1. 按输出 shape 自动分组，不盲信 output 顺序。
2. cls 分支已经是 sigmoid 后概率，不再 sigmoid。
3. 用 max(cls) >= conf_thres 先筛 anchor。
4. 只对保留下来的 anchor 做 DFL decode。
5. det_score = max(cls)，class_id = argmax(cls)。
6. 按类别做 NMS。
7. 用 letterbox 的 ratio/pad 还原原图坐标。
```

注意：

```text
cls_sum 只是 cls 求和再 clamp 到 0~1 的辅助过滤信号。
它不是 YOLOv5 那种 objectness。
当前脚本最终检测分数不用 cls_sum。
```

### 7.2 LPRNet 后处理

LPRNet 输出：

```text
logits = (1,74,18)
```

脚本会兼容：

```text
(1,74,18)
(74,18)
(1,18,74)
(18,74)
```

但最终都会整理成：

```text
(74,18)
```

识别逻辑：

```text
1. logits -> log_softmax。
2. 每个时间步取 topk 字符和 blank。
3. CTC beam search 保留多个候选路径。
4. 根据 plate_type 限制长度和字符位置。
5. 优先选择长度合法、格式合法、beam score 最高的结果。
```

当前字符表：

```text
31 省份
10 数字
24 字母，去掉 I/O
8 特殊字符: 学/挂/港/澳/使/领/警/临
1 blank: -
```

关键约束：

```text
blue/yellow/black/special_7:
  目标长度 7

green/unknown_8:
  目标长度 8

普通车牌:
  第 1 位省份
  第 2 位字母
  后续字母或数字

yellow:
  后续允许 学/挂

special_7:
  后续允许 学/挂/港/澳/使/领/警/临

black:
  当前规则较宽，允许非 blank 字符
```

## 8. 常见问题

### 8.1 报错：core_mask is only supported by ['RK3588', 'RK3576']

如果在 RK3568 上看到类似报错：

```text
E The core_mask is only supported by ['RK3588', 'RK3576'].
E Catch exception when set npu core mode.
RuntimeError: init_runtime failed: fenqusai/rknn/best.rknn
```

说明命令里设置了 `--core-mask core012`、`--core-mask core0` 或其他手动绑核参数。

RK3568 运行时请显式加上：

```bash
--core-mask default --lpr-core-mask default
```

完整示例：

```bash
python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image 'fenqusai/test_photo/ccpd2020_test__055373563218390806-90_250-93_478_585_581-585_581_125_574_93_478_582_480-0_0_3_32_29_30_29_32-108-249.jpg' \
  --output-dir fenqusai/tools/pipeline/result_rknn_plate \
  --core-mask default \
  --lpr-core-mask default
```

只有 RK3588/RK3576 才建议尝试：

```bash
--core-mask core012
```

### 8.2 报错：YOLO RKNN should return 9 outputs

说明当前 `--yolo-model` 不是 RKNN 友好的 9 输出 YOLO，可能是普通 YOLO ONNX/RKNN 或带 NMS 的模型。

需要确认：

```text
best.rknn 输出是否仍是 9 outputs
是否拿错了模型文件
是否使用了带 NMS 的导出版本
```

### 8.3 检测框位置整体偏移

优先检查：

```text
letterbox 是否一致
padding 是否用 114
坐标还原是否减 pad 再除 ratio
```

当前脚本已经记录并使用 `ratio/pad` 还原坐标。

### 8.4 检测不到车牌

排查顺序：

```text
1. 降低 --conf-thres。
2. 切换 --yolo-color rgb/bgr。
3. 确认 best.rknn 的类别顺序是 plate/person/car/traffic_light。
4. 打印 RKNN output shape，确认 9 输出形状符合预期。
```

### 8.5 检测框正确，但车牌识别乱码

排查顺序：

```text
1. 确认 LPRNet 输出 class dim 是 74。
2. 确认 CHARS 顺序和训练时完全一致。
3. 切换 --lpr-color bgr/rgb。
4. 检查 crop 是否太紧或背景太多。
5. 确认没有重复归一化。
```

### 8.6 raw_text 对，plate_text 不对

`raw_text` 是 greedy 调试输出，`plate_text` 是 constrained decode 输出。

如果 raw_text 更像正确结果，而 constrained 输出不对，通常说明：

```text
plate_type 粗判错误
格式约束过强
beam_width/topk 太小
```

可以尝试：

```bash
--beam-width 20 --beam-topk 10
```

或临时修改 `decode_with_type_fallback()` 的尝试顺序。

### 8.7 plate_text 对，lpr_score 很低

`lpr_score` 是 beam log score 按时间步归一化后的粗略概率，不是严格业务置信度。

第一版不要只靠它删除结果。建议同时看：

```text
det_score
plate_text 是否长度合法
可视化 crop 是否清晰
beam_score 排名
```

## 9. 建议验证流程

第一轮建议选 10 张固定图片：

```text
单蓝牌
单绿牌
单黄牌
黑牌
双层黄牌
拖拉机绿牌
多车牌
夜间/模糊
小目标车牌
含 person/car/traffic_light 的复杂场景
```

每张图记录：

```text
1. YOLO detections JSON
2. plate crop
3. raw_text
4. plate_text
5. plate_type
6. 可视化图片
```

推荐命令：

```bash
python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image test_photo/Random \
  --yolo-model fenqusai/rknn/best.rknn \
  --lpr-model fenqusai/rknn/lprnet_unified_p15_focus_fp.rknn \
  --output-dir fenqusai/tools/pipeline/result_p15_fp \
  --core-mask default \
  --lpr-core-mask default \
  --beam-width 10 \
  --beam-topk 8
```

FP 跑通后，再对比 INT8：

```bash
python3 fenqusai/tools/pipeline/infer_rknn_plate.py \
  --image test_photo/Random \
  --lpr-model fenqusai/rknn/lprnet_unified_p15_focus_i8.rknn \
  --output-dir fenqusai/tools/pipeline/result_p15_i8 \
  --core-mask default \
  --lpr-core-mask default
```

如果 INT8 识别省份、特殊牌明显掉点，最终部署优先保留 FP LPRNet。

## 10. 一句话结论

这个脚本的目标不是极限性能，而是先把当前 RKNN 模型的完整后处理链路跑准：

```text
YOLO 9 outputs
  -> fast DFL decode
  -> class-wise NMS
  -> plate crop
  -> LPRNet logits
  -> constrained CTC beam search
```

等图片验证稳定后，再把同一套逻辑迁移到视频流或 C++ RKNN C API。

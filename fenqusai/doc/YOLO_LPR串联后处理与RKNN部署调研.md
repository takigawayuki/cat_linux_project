# YOLO + LPRNet 串联后处理与 RKNN 部署调研

本文用于解释当前项目里“YOLO 检测 + LPRNet 识别”在瑞芯微 RKNN 板端部署时，模型输出之后到底要做什么。重点是给后续 AI 或工程师继续写 C++/Python 板端代码时使用。

## 1. 当前项目状态

### 1.1 YOLO 检测模型

当前 YOLO 数据集配置：

```text
datasets/yolo_lite/yolo.yaml

nc: 4
names:
  0: plate
  1: person
  2: car
  3: traffic_light
```

当前推荐检测权重：

```text
runs/yolo4_lite_n_50e_local_b8_amp_v2/weights/best.pt
```

已经讨论过的 RKNN 友好导出方式：

```powershell
yolo export model=runs\yolo4_lite_n_50e_local_b8_amp_v2\weights\best.pt format=rknn
```

导出后的 YOLO RKNN 友好 ONNX 不再是普通 YOLOv8 的一个大输出，而是 3 个尺度，每个尺度 3 个输出，共 9 个输出：

```text
scale 80x80:
  bbox:      (1, 64, 80, 80)
  cls:       (1, 4, 80, 80)
  cls_sum:   (1, 1, 80, 80)

scale 40x40:
  bbox:      (1, 64, 40, 40)
  cls:       (1, 4, 40, 40)
  cls_sum:   (1, 1, 40, 40)

scale 20x20:
  bbox:      (1, 64, 20, 20)
  cls:       (1, 4, 20, 20)
  cls_sum:   (1, 1, 20, 20)
```

其中 `bbox` 是 DFL 回归分支，`cls` 是各类别置信度，`cls_sum` 是类别置信度求和后裁剪到 0~1 的 objectness-like 分支。

本项目源码对应位置：

```text
ultralytics_yolov8/ultralytics/nn/modules/head.py
```

核心逻辑：

```python
if self.export and self.format == 'rknn':
    y = []
    for i in range(self.nl):
        y.append(self.cv2[i](x[i]))
        cls = torch.sigmoid(self.cv3[i](x[i]))
        cls_sum = torch.clamp(cls.sum(1, keepdim=True), 0, 1)
        y.append(cls)
        y.append(cls_sum)
    return y
```

当前已直接检查：

```text
runs/yolo4_lite_n_50e_local_b8_amp_v2/weights/best.onnx
```

检查结果：

```text
outputs: 9
output names:
  conv2d_47, sigmoid, clamp
  conv2d_53, sigmoid_1, clamp_1
  conv2d_59, sigmoid_2, clamp_2

output shapes:
  (1, 64, 80, 80), (1, 4, 80, 80), (1, 1, 80, 80)
  (1, 64, 40, 40), (1, 4, 40, 40), (1, 1, 40, 40)
  (1, 64, 20, 20), (1, 4, 20, 20), (1, 1, 20, 20)

NonMaxSuppression nodes: 0
```

结论：

```text
当前 YOLO ONNX 没有内置 NMS。
它只输出 RKNN 友好的原始检测头结果。
后续必须在 CPU 后处理里做 DFL decode、阈值过滤、NMS 和坐标还原。
```

### 1.2 LPRNet 识别模型

当前推荐主模型：

```text
LPRNet_Pytorch/weights/unified_p15_focus/Final_LPRNet_model.pth
LPRNet_Pytorch/weights/unified_p15_focus/lprnet_unified_p15_focus_sim.onnx
```

当前备选折中模型：

```text
LPRNet_Pytorch/weights/unified_p2_province_focus/Final_LPRNet_model.pth
LPRNet_Pytorch/weights/unified_p2_province_focus/lprnet_unified_p2_province_focus_sim.onnx
```

当前 LPRNet ONNX 输入输出：

```text
input:  (1, 3, 24, 94)
output: (1, 74, 18)
```

`74` 是字符类别数，来自：

```text
LPRNet_Pytorch/data/load_data.py
CHARS
```

组成：

```text
31 个省份简称
10 个数字
24 个字母，去掉 I/O
8 个特殊字符：学/挂/港/澳/使/领/警/临
1 个 CTC blank：-
```

`18` 是 CTC 时间步，不是车牌长度。也就是说，LPRNet 不是直接输出“粤B12345”这种字符串，而是输出 18 个时间步上 74 个字符类别的分数。

## 2. “ONNX 只输出 logits”是什么意思

LPRNet 输出 `(1, 74, 18)`，可以理解为：

```text
batch=1
class_num=74
time_steps=18
```

去掉 batch 后就是：

```text
logits: (74, 18)
```

每个时间步 `t` 上都有 74 个原始分数：

```text
logits[:, t] = 第 t 个时间步对 74 个字符的打分
```

这些分数还不是最终字符，也不是概率。后处理需要做：

```text
1. 对每个时间步做 softmax 或直接 argmax。
2. 得到长度为 18 的原始字符 index 序列。
3. 按 CTC 规则去掉重复字符和 blank。
4. 得到初步车牌字符串。
5. 再结合车牌类型做 constrained decode / 格式约束。
```

所以“CTC 解码和 constrained decode 放在板端 CPU 后处理做”的意思是：

```text
NPU/RKNN:
  只跑 LPRNet CNN 前向，输出 logits。

CPU:
  读取 logits。
  做 CTC greedy 或 beam search。
  根据车牌类型约束字符合法性。
  输出最终车牌字符串。
```

这样做更适合瑞芯微部署，因为 NPU 擅长 Conv/ReLU/Pool 这类神经网络算子，而 CTC 解码、beam search、字符串规则判断、NMS、坐标还原这些逻辑分支更适合 CPU。

瑞芯微官方 LPRNet 示例也是这个思路：RKNN 模型跑完后，demo 输出“车牌识别结果”，也就是模型前向之后仍有一层识别结果解析逻辑。官方 LPRNet 示例支持 RK3562/RK3566/RK3568/RK3576/RK3588 等平台，并使用 `convert.py <onnx_model> <TARGET_PLATFORM>` 将 ONNX 转 RKNN。

## 3. LPRNet 后处理：从 logits 到车牌字符串

### 3.1 输入预处理必须和训练一致

当前训练 DataLoader 的预处理在：

```text
LPRNet_Pytorch/data/load_data.py
```

代码逻辑：

```python
img = cv2.resize(img, (94, 24))
img = img.astype("float32")
img -= 127.5
img *= 0.0078125
img = np.transpose(img, (2, 0, 1))
```

等价于：

```text
resize 到 94x24
BGR 输入
归一化: (img - 127.5) / 128
HWC -> CHW
```

RKNN 转换建议：

```python
rknn.config(
    mean_values=[[127.5, 127.5, 127.5]],
    std_values=[[128.0, 128.0, 128.0]],
    target_platform="rk3588",
)
```

如果板端 C++ 自己做归一化，则 RKNN config 里不要重复归一化；如果 RKNN config 做归一化，则板端只传 uint8 BGR 输入。二者只能保留一套，否则会归一化两次。

### 3.2 CTC greedy decode

当前项目已有 greedy decode 参考：

```text
LPRNet_Pytorch/test_LPRNet.py
greedy_decode_one(preb)
```

核心步骤：

```text
1. 对每个时间步取 argmax：
   raw[t] = argmax(logits[:, t])

2. 遍历 raw：
   如果当前字符是 blank，跳过。
   如果当前字符和上一个非 blank 字符重复，跳过。
   否则加入结果。
```

伪代码：

```python
blank = len(CHARS) - 1
raw = argmax(logits, axis=0)  # shape=(18,)

result = []
prev = None
for c in raw:
    if c == blank:
        prev = c
        continue
    if c != prev:
        result.append(c)
    prev = c

text = "".join(CHARS[i] for i in result)
```

注意：CTC 的“去重”只合并连续重复。比如：

```text
原始时间步: 粤 粤 - B B 1 - 2 3 4 5
CTC 后:    粤 B 1 2 3 4 5
```

如果真实车牌中确实有重复字符，例如 `粤B11111`，模型需要在重复字符之间输出 blank 才能区分：

```text
1 - 1 - 1
```

### 3.3 constrained CTC decode

当前项目推荐使用 constrained decode，而不是纯 greedy。对应代码：

```text
LPRNet_Pytorch/test_LPRNet.py
constrained_ctc_decode_one(...)
```

它比 greedy 多做两件事：

```text
1. 用 beam search 保留多个候选路径，而不是每个时间步只取最大值。
2. 用车牌类型规则过滤非法前缀。
```

当前规则入口：

```text
target_len_for_type(plate_type, plate_subtype)
is_valid_prefix(prefix, plate_type, plate_subtype)
```

长度规则：

```text
blue:       7
yellow:     7
black:      7
unknown_7:  7
special_7:  7
green:      8
unknown_8:  8
```

字符位置规则：

```text
普通 blue/green/yellow:
  第 1 位: 省份简称
  第 2 位: 字母
  第 3 位以后: 字母或数字

yellow:
  第 3 位以后允许部分特殊字符，例如 学/挂

tractor_green:
  第 1 位: 省份简称
  第 2 位以后: 字母/数字/部分特殊字符

special_7:
  第 1 位: 省份简称
  第 2 位: 字母或数字
  第 3 位以后: 字母/数字/特殊字符

black:
  当前放宽为任意非 blank 字符
```

beam search 解码大致流程：

```text
1. logits -> log_probs。
2. beams 初始为一个空前缀。
3. 每个时间步只取 topk 字符和 blank，减少计算量。
4. 对每个候选前缀扩展新字符。
5. 扩展后调用 is_valid_prefix，不合法就丢弃。
6. 每个时间步只保留 beam_width 个最高分候选。
7. 最后优先选择长度等于目标长度的候选。
```

当前测试命令里常用参数：

```powershell
--decode_mode constrained --beam_width 10 --beam_topk 8
```

板端建议：

```text
先实现 greedy decode，验证 RKNN 输出和 Python 能对上。
再实现 constrained beam search。
最后把 beam_width/topk 做成可配置项。
```

### 3.4 车牌类型从哪里来

LPRNet 识别本身只输出字符，不输出颜色或类型。板端 constrained decode 需要知道 `plate_type`，来源有三个选择：

```text
方案 A：由 YOLO 检测类别提供。
  如果 YOLO 只检测 plate/person/car/traffic_light，那么只能知道它是 plate，不知道蓝/绿/黄/黑。

方案 B：由车牌 crop 的颜色规则判断。
  对 crop 做 HSV 颜色统计，判断 green/yellow/blue/black。

方案 C：让 YOLO 检测类别细分为 blue_plate/green_plate/yellow_plate/black_plate。
  当前项目 YOLO 不是这么训练的，因此暂时不能直接用。
```

当前更现实的做法：

```text
1. YOLO 检测出 plate。
2. 对 plate crop 做颜色粗判。
3. 根据颜色选择 plate_type。
4. LPRNet 输出 logits。
5. 使用对应 plate_type 做 constrained decode。
```

颜色粗判可以先用简单 HSV：

```text
green: 绿色像素占比高
yellow: 黄色像素占比高
blue: 蓝色像素占比高
black: 整体亮度较低且饱和度/暗色区域符合
```

如果颜色判断不确定：

```text
同时用多个 plate_type 跑 constrained decode，取分数最高且格式合法的结果。
```

这比强行相信一个颜色判断更稳。

## 4. YOLO RKNN 后处理：从 9 输出到检测框

### 4.1 为什么 YOLO RKNN 是 9 输出

瑞芯微官方 YOLOv8 示例说明，优化版模型会把原始一个输出拆成 3 组输出。以 80x80 尺度为例：

```text
[1,64,80,80]  bbox / DFL 回归
[1,80,80,80]  80 类置信度
[1,1,80,80]   80 类置信度求和
```

当前项目是 4 类，所以对应变成：

```text
[1,64,80,80]
[1,4,80,80]
[1,1,80,80]
```

瑞芯微这样做的原因是：把 DFL 解码、网格拼接、NMS 等复杂后处理移到 CPU，NPU 只负责卷积输出。这样更适合 RKNN 工具链，也更方便 C API demo 做统一后处理。

### 4.2 YOLO 输入预处理

当前 ONNX 测试脚本参考：

```text
tools/test/detect_onnx_pipeline.py
preprocess_yolo(img)
```

逻辑：

```text
1. letterbox 到 640x640，填充值 114。
2. BGR/RGB 顺序必须和导出/训练保持一致。
3. 转 float32。
4. 除以 255。
5. HWC -> CHW。
6. 增加 batch 维度。
```

伪代码：

```python
img_640, scale, left, top = letterbox(img, 640)
inp = img_640.astype(np.float32) / 255.0
inp = inp.transpose(2, 0, 1)[None]
```

RKNN 侧可以选择：

```text
方案 A：板端 CPU 做 /255，RKNN 输入 float。
方案 B：RKNN config 设置 mean/std，板端输入 uint8。
```

YOLO 部署中常见做法是板端传 uint8，RKNN config 里处理均值方差；但要以转换脚本和实际 demo 为准，不能重复归一化。

### 4.3 YOLO 9 输出后处理步骤

输入：

```text
outputs = [
  bbox_80, cls_80, score_80,
  bbox_40, cls_40, score_40,
  bbox_20, cls_20, score_20,
]
```

每个尺度：

```text
bbox:  (1, 64, H, W)
cls:   (1, 4, H, W)
score: (1, 1, H, W)
```

后处理流程：

```text
1. 遍历 3 个尺度。
2. 用 score 或 max(cls) 做置信度预过滤。
3. 对 bbox 的 64 通道做 DFL 解码：
   64 = 4 * reg_max，reg_max=16。
4. 根据 grid cell 和 stride 还原到 640x640 letterbox 坐标。
5. 取类别 class_id 和 class_score。
6. 过滤 conf_thres 以下的框。
7. 按类别或全局做 NMS。
8. 把 letterbox 坐标映射回原图坐标。
```

DFL 解码概念：

```text
bbox 每个边距离不是直接回归一个数，而是 16 个 bin 的分布。
对每条边的 16 个分数做 softmax。
用 0..15 加权求和，得到 left/top/right/bottom 距离。
```

伪代码：

```python
proj = np.arange(16)
dist = bbox.reshape(4, 16, H, W)
dist = softmax(dist, axis=1)
dist = (dist * proj.reshape(1, 16, 1, 1)).sum(axis=1)
```

坐标还原：

```text
grid_x, grid_y 是当前 cell 坐标。
stride 分别是 8 / 16 / 32。

x1 = (grid_x + 0.5 - left_dist) * stride
y1 = (grid_y + 0.5 - top_dist) * stride
x2 = (grid_x + 0.5 + right_dist) * stride
y2 = (grid_y + 0.5 + bottom_dist) * stride
```

再从 640x640 letterbox 坐标还原原图：

```python
x1 = (x1 - left_pad) / scale
x2 = (x2 - left_pad) / scale
y1 = (y1 - top_pad) / scale
y2 = (y2 - top_pad) / scale
```

最后 clip 到原图范围。

### 4.4 当前项目类别的后处理选择

当前类别：

```text
0 plate
1 person
2 car
3 traffic_light
```

串联 LPR 时只把 `class_id == 0` 的框送给 LPRNet。

```text
plate:
  进入 LPRNet crop + 识别。

person/car/traffic_light:
  保留给后续行人违章、车辆、红绿灯状态逻辑。
```

NMS 建议：

```text
1. 对所有类别一起 NMS 或按类别 NMS 都可以。
2. 如果 plate 和 car 框重叠很大，建议按类别 NMS，避免 car 把 plate 抑制掉。
3. plate 类可以设置更低一点的 conf_thres，例如 0.25~0.35。
4. person/car/traffic_light 可以按后续任务需要单独设阈值。
```

## 5. YOLO + LPRNet 串联流程

### 5.1 总流程

```text
camera frame / image
  -> YOLO preprocess
  -> RKNN YOLO inference on NPU
  -> YOLO CPU postprocess: DFL decode + NMS + scale back
  -> filter class_id == plate
  -> crop plate region from original image
  -> optional crop expand
  -> LPRNet preprocess: resize 94x24 + normalize
  -> RKNN LPRNet inference on NPU
  -> LPRNet CPU postprocess: CTC decode / constrained decode
  -> final plate text + confidence
```

### 5.2 裁剪策略

YOLO 检测框直接裁剪可能会偏紧，建议给 plate 框加一点 padding：

```text
水平 padding: 5%~10%
垂直 padding: 10%~20%
```

原因：

```text
1. LPRNet 需要完整字符边缘。
2. 双层黄牌和拖拉机绿牌如果裁剪太紧，第一位省份更容易错。
3. YOLO 框轻微偏移时，padding 能提高容错。
```

裁剪后至少做：

```text
1. clip 到原图范围。
2. 检查 crop 宽高大于最小值。
3. 宽高比异常的 crop 降低置信度或跳过。
```

建议最小尺寸：

```text
crop width >= 20
crop height >= 8
```

### 5.3 当前不做透视矫正

当前 pipeline 明确不加入透视矫正，直接使用：

```text
YOLO axis-aligned box crop -> resize 94x24 -> LPRNet
```

原因：

```text
1. 当前 YOLO 检测模型输出的是水平框，不输出四点角点。
2. 强行做透视矫正需要额外角点估计，工程复杂度会明显增加。
3. 当前 LPRNet 训练数据就是裁剪后 resize 到 94x24 的识别方式，直接 crop 更贴近现有训练链路。
4. 第一版 RKNN 串联重点是稳定跑通 YOLO 检测框、LPRNet logits、CTC/constrained 后处理。
```

后续任务描述里不要再要求实现透视矫正。

### 5.4 多车牌场景

多车牌场景不要让 LPRNet 一次识别整张图。正确方式：

```text
1. YOLO 检出多个 plate 框。
2. 每个 plate 框单独裁剪。
3. 每个 crop 单独送入 LPRNet。
4. 输出多个车牌结果。
```

排序建议：

```text
按 y1 从上到下，再按 x1 从左到右排序。
```

输出结构建议：

```json
{
  "bbox": [x1, y1, x2, y2],
  "det_score": 0.91,
  "plate_type": "green",
  "plate_text": "粤BD12345",
  "lpr_score": 0.86,
  "decode_mode": "constrained"
}
```

### 5.5 plate_type 判定建议

因为当前 YOLO 只有 `plate` 类，不细分颜色，所以板端要补一个 `estimate_plate_type(crop)`：

```text
输入: plate crop BGR
输出: blue / green / yellow / black / unknown
```

简单规则：

```python
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

blue_ratio = count HSV in blue range / area
green_ratio = count HSV in green range / area
yellow_ratio = count HSV in yellow range / area
dark_ratio = count low V pixels / area

if green_ratio > threshold:
    return "green"
if yellow_ratio > threshold:
    return "yellow"
if blue_ratio > threshold:
    return "blue"
if dark_ratio > threshold:
    return "black"
return "unknown_7"
```

如果 `unknown`：

```text
1. 同时尝试 blue/yellow/black/green 四种 constrained decode。
2. 每种 decode 返回候选 text 和 beam score。
3. 优先选格式合法、长度合理、beam score 高的结果。
```

### 5.6 LPR 置信度建议

greedy decode 可以粗略取：

```text
输出字符对应时间步最大 softmax 概率的平均值
```

constrained beam search 可以取：

```text
最终候选 prefix 的 log probability
```

为了和检测置信度结合，建议归一化成：

```text
final_score = det_score * lpr_score
```

用于排序和过滤，但不要只靠这个分数删除车牌。车牌识别场景里，低置信度结果也可能对调试有价值。

## 6. 瑞芯微 RKNN 部署时的分工

### 6.1 NPU 做什么

```text
YOLO RKNN:
  输入 640x640 图像。
  输出 9 个 feature map。

LPRNet RKNN:
  输入 94x24 车牌 crop。
  输出 logits=(1,74,18)。
```

### 6.2 CPU 做什么

```text
YOLO:
  letterbox 参数记录
  DFL decode
  conf filter
  NMS
  坐标还原

LPRNet:
  CTC greedy decode
  constrained beam search
  车牌格式约束
  CHARS index -> 字符串
  多车牌结果排序
```

### 6.3 为什么不要把后处理塞进 ONNX/RKNN

不建议把 NMS、CTC decode、字符串判断塞进模型图，原因：

```text
1. RKNN 对动态循环、字符串逻辑、复杂分支支持不适合。
2. NMS/beam search 在 CPU 上更容易写、调试、验证。
3. 后处理规则后续还会频繁改，放 CPU 不需要重新转模型。
4. 瑞芯微官方 YOLOv8 优化模型也是把输出拆开，把复杂后处理留给 demo 代码。
```

## 7. 后续代码实现建议

### 7.1 建议新增模块

建议后续新建：

```text
pipeline/rknn_yolo_postprocess.py 或 C++ 对应文件
pipeline/rknn_lpr_postprocess.py 或 C++ 对应文件
pipeline/plate_pipeline.py 或 C++ 对应文件
```

如果直接写 C++，建议函数拆成：

```text
YOLO:
  letterbox()
  decode_yolov8_rknn_outputs()
  dfl_decode()
  nms()
  scale_boxes_to_original()

LPRNet:
  preprocess_lpr_crop()
  softmax_per_timestep()
  ctc_greedy_decode()
  constrained_ctc_beam_decode()
  estimate_plate_type()
  chars_index_to_utf8()

Pipeline:
  run_yolo()
  run_lpr_for_plate_boxes()
  render_results()
```

### 7.2 第一版最小闭环

第一版不要一口气做全部规则，建议顺序：

```text
1. 用 RKNN YOLO 跑通图片输入，拿到 plate 框。
2. 只对 plate 框 crop，不做透视矫正。
3. 用 RKNN LPRNet 跑 crop，拿到 logits。
4. 先 greedy decode 输出字符串。
5. 对比 Python ONNX / PyTorch 的同图结果。
6. 再加 constrained decode。
7. 最后加颜色判断和多车牌排序。
```

### 7.3 验证方法

建议准备 10 张固定测试图：

```text
单蓝牌
单绿牌
单黄牌
单黑牌
双层黄牌
拖拉机绿牌
多车牌 CRPD_multi
夜间/模糊图
小目标车牌
含行人/车/红绿灯场景
```

每张图记录：

```text
YOLO bbox
det_score
crop 保存路径
plate_type 判断结果
LPR raw argmax 序列
greedy text
constrained text
final text
```

这样后续如果 RKNN 输出和 Python 不一致，可以定位到底是：

```text
YOLO 前处理问题
YOLO 后处理问题
crop 问题
LPR 前处理问题
LPR CTC decode 问题
CHARS 映射问题
```

## 8. 当前最容易踩的坑

### 8.1 CHARS 编码必须完全一致

板端 C++ 里的字符表必须和：

```text
LPRNet_Pytorch/data/load_data.py
```

完全一致，顺序也必须一致。不能只写“省份 + 数字 + 字母”然后自己随便排序。

如果顺序错了，模型输出 index 没变，但映射出来的字符会全错。

### 8.2 blank index 是 73

当前：

```text
class_num = 74
blank_index = 73
blank_char = "-"
```

CTC decode 时必须跳过 index 73。

### 8.3 LPRNet 输出不是概率

输出是 logits。可以：

```text
greedy: 直接 argmax logits
beam search: 先转 log_softmax
```

不要把 `(1,74,18)` 当成 18 个字符直接查表。

### 8.4 YOLO 的普通 ONNX 和 RKNN 友好 ONNX 后处理不同

普通 YOLO ONNX 可能是：

```text
output0: (1, 4+nc, 8400)
```

RKNN 友好 YOLO ONNX 是：

```text
9 outputs
```

两种后处理完全不同。板端接瑞芯微 `rknn_model_zoo/examples/yolov8` 风格代码时，应使用 9 输出版本。

### 8.5 坐标还原必须用 letterbox 参数

不能直接把 640 坐标按原图宽高比例缩放。必须减掉 padding：

```text
x = (x - left_pad) / scale
y = (y - top_pad) / scale
```

否则车牌 crop 会偏移，LPRNet 精度会明显下降。

## 9. 参考资料

瑞芯微官方：

```text
https://github.com/airockchip/rknn_model_zoo
https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8
https://github.com/airockchip/rknn_model_zoo/tree/main/examples/LPRNet
https://github.com/airockchip/rknn-toolkit2
https://github.com/airockchip/ultralytics_yolov8
```

用户提供的 YOLOv8 RK3588 多线程部署教程：

```text
https://github.com/JA-cmd-wq/yolov8-helmet-rk3588-multithread/blob/main/docs/TUTORIAL.md
```

本项目相关文件：

```text
ultralytics_yolov8/ultralytics/nn/modules/head.py
LPRNet_Pytorch/data/load_data.py
LPRNet_Pytorch/test_LPRNet.py
LPRNet_Pytorch/export_unified_lprnet_onnx.py
tools/test/detect_onnx_pipeline.py
doc/pipeline/LPRNet_ONNX导出适配RKNN方案.md
doc/pipeline/YOLO检测模型导出RKNN说明.md
```

## 10. 给后续 AI/工程师的直接任务

如果后续要根据本文继续写代码，建议任务描述可以这样写：

```text
请基于当前项目的 YOLO RKNN 9 输出和 LPRNet logits=(1,74,18) 输出，实现 RK3588 板端 YOLO+LPRNet 串联后处理。

要求：
1. YOLO 使用 640 letterbox 输入，解析 3 个尺度共 9 个输出。
2. 当前 YOLO ONNX 没有内置 NMS，后处理必须包含 DFL decode、阈值过滤、按类别 NMS、坐标还原。
3. 只把 class_id=0 的 plate 框裁剪送入 LPRNet。
4. LPR crop resize 到 94x24，保持 BGR，归一化与训练一致。
5. LPRNet 输出 logits=(1,74,18)，实现 greedy CTC decode 和 constrained beam search decode。
6. CHARS 顺序必须与 LPRNet_Pytorch/data/load_data.py 完全一致，blank index=73。
7. constrained decode 根据 plate_type 限制长度和字符位置。
8. plate_type 暂时由 crop HSV 颜色粗判；不确定时多规则解码取最佳候选。
9. 输出每个车牌的 bbox、det_score、plate_type、plate_text、lpr_score。
10. 代码结构要方便以后替换成 C++ RKNN C API。
11. 当前版本不做透视矫正，不需要角点检测或四点变换。
```

## 11. 交接总览：当前已有内容和后续目标

这一节是给后续 AI / 工程师的快速交接摘要。后续如果只想先判断项目进度，可以先读这一节，再回看前面的细节。

### 11.1 当前已经有的东西

YOLO 检测侧：

```text
模型权重:
  runs/yolo4_lite_n_50e_local_b8_amp_v2/weights/best.pt

RKNN 友好 ONNX:
  runs/yolo4_lite_n_50e_local_b8_amp_v2/weights/best.onnx
  runs/yolo4_lite_n_50e_local_b8_amp_v2/weights/best.onnx.data

类别:
  0: plate
  1: person
  2: car
  3: traffic_light

ONNX 输出:
  9 outputs
  无内置 NMS
  需要 CPU 做 DFL decode / conf filter / NMS / scale back
```

LPRNet 识别侧：

```text
主模型:
  LPRNet_Pytorch/weights/unified_p15_focus/Final_LPRNet_model.pth
  LPRNet_Pytorch/weights/unified_p15_focus/lprnet_unified_p15_focus.onnx
  LPRNet_Pytorch/weights/unified_p15_focus/lprnet_unified_p15_focus_sim.onnx

备选折中模型:
  LPRNet_Pytorch/weights/unified_p2_province_focus/Final_LPRNet_model.pth
  LPRNet_Pytorch/weights/unified_p2_province_focus/lprnet_unified_p2_province_focus.onnx
  LPRNet_Pytorch/weights/unified_p2_province_focus/lprnet_unified_p2_province_focus_sim.onnx

ONNX 输入:
  (1, 3, 24, 94)

ONNX 输出:
  logits=(1, 74, 18)

后处理:
  CPU 做 CTC greedy decode / constrained beam search decode
```

LPRNet 导出相关代码：

```text
LPRNet_Pytorch/model/LPRNet_export.py
LPRNet_Pytorch/export_unified_lprnet_onnx.py
```

LPRNet 后处理参考代码：

```text
LPRNet_Pytorch/test_LPRNet.py
  greedy_decode_one
  constrained_ctc_decode_one
  target_len_for_type
  is_valid_prefix
```

### 11.2 后续要做的事情

后续还没有完成的核心任务：

```text
1. 把 YOLO best.onnx 转成 yolov8_plate.rknn。
2. 把 LPRNet p15_focus sim ONNX 转成 lprnet_p15_focus.rknn。
3. 可选：把 LPRNet p2_province_focus sim ONNX 也转成 lprnet_p2_province_focus.rknn。
4. 写 RKNN Python 或 C++ pipeline：
   YOLO RKNN -> YOLO 后处理 -> plate crop -> LPRNet RKNN -> LPR 后处理。
5. 先在 PC / 板端 Python 验证，再移植到 C++ RKNN C API。
6. 后续不做透视矫正，只做水平框 crop + padding + resize。
```

### 11.3 当前建议的模型选择

默认主线：

```text
YOLO:
  runs/yolo4_lite_n_50e_local_b8_amp_v2/weights/best.onnx

LPRNet:
  LPRNet_Pytorch/weights/unified_p15_focus/lprnet_unified_p15_focus_sim.onnx
```

备选 LPR：

```text
LPRNet_Pytorch/weights/unified_p2_province_focus/lprnet_unified_p2_province_focus_sim.onnx
```

选择原因：

```text
p15_focus:
  主测试集最稳，适合作为默认最终 LPR 模型。

p2_province_focus:
  稍微照顾 CRPD / 多车牌裁剪风格，但不是默认替代 p15_focus。
```

## 12. ONNX 转 RKNN：YOLO 和 LPRNet 分别怎么转

### 12.1 瑞芯微官方示例的转换方式

瑞芯微 `rknn_model_zoo` 的 YOLOv8 和 LPRNet 示例都是类似命令：

```shell
cd examples/yolov8/python
python convert.py ../model/yolov8n.onnx rk3588
```

```shell
cd examples/LPRNet/python
python convert.py ../model/lprnet.onnx rk3588
```

官方参数含义：

```text
第 1 个参数: ONNX 模型路径
第 2 个参数: target platform，例如 rk3588
第 3 个参数: 可选，i8/u8/fp
  i8 / u8: 做量化
  fp: 不量化
第 4 个参数: 可选，输出 RKNN 路径
```

本项目后续可以直接复用官方 convert.py，也可以自己写一个项目内转换脚本。建议第一版先复用官方脚本，减少变量。

参考：

```text
https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8
https://github.com/airockchip/rknn_model_zoo/tree/main/examples/LPRNet
```

### 12.2 YOLO 转 RKNN 建议

输入 ONNX：

```text
runs/yolo4_lite_n_50e_local_b8_amp_v2/weights/best.onnx
```

注意：当前 YOLO ONNX 有外部数据文件：

```text
runs/yolo4_lite_n_50e_local_b8_amp_v2/weights/best.onnx.data
```

复制、移动或上传到转换环境时，必须把下面两个文件放在同一目录：

```text
best.onnx
best.onnx.data
```

建议输出：

```text
deploy/rknn/yolov8_plate_4cls.rknn
```

建议先转 FP 版本验证后处理：

```shell
python convert.py best.onnx rk3588 fp yolov8_plate_4cls_fp.rknn
```

FP 版本跑通后，再考虑 INT8 / U8 量化：

```shell
python convert.py best.onnx rk3588 i8 yolov8_plate_4cls_i8.rknn
```

量化注意：

```text
1. INT8 需要 calibration dataset。
2. calibration 图片要覆盖 plate/person/car/traffic_light。
3. 如果只用少量单一车牌图量化，person/car/traffic_light 可能掉点。
4. 第一版 pipeline 验证优先用 fp，确认后处理正确后再量化。
```

YOLO RKNN 输出仍应是 9 outputs。转换后要打印 output attr，确认输出数量和形状仍是：

```text
(1, 64, 80, 80), (1, 4, 80, 80), (1, 1, 80, 80)
(1, 64, 40, 40), (1, 4, 40, 40), (1, 1, 40, 40)
(1, 64, 20, 20), (1, 4, 20, 20), (1, 1, 20, 20)
```

如果 RKNN 输出顺序不同，后处理必须按实际 output attr 重排，不要盲目按文件名猜。

### 12.3 LPRNet 转 RKNN 建议

默认输入 ONNX：

```text
LPRNet_Pytorch/weights/unified_p15_focus/lprnet_unified_p15_focus_sim.onnx
```

备选输入 ONNX：

```text
LPRNet_Pytorch/weights/unified_p2_province_focus/lprnet_unified_p2_province_focus_sim.onnx
```

建议输出：

```text
deploy/rknn/lprnet_unified_p15_focus_fp.rknn
deploy/rknn/lprnet_unified_p2_province_focus_fp.rknn
```

建议先转 FP：

```shell
python convert.py lprnet_unified_p15_focus_sim.onnx rk3588 fp lprnet_unified_p15_focus_fp.rknn
```

再考虑量化：

```shell
python convert.py lprnet_unified_p15_focus_sim.onnx rk3588 i8 lprnet_unified_p15_focus_i8.rknn
```

LPRNet 量化注意：

```text
1. LPRNet 对字符分类较敏感，INT8 可能影响省份简称和特殊牌。
2. calibration crop 必须覆盖 blue/green/yellow/black。
3. 尤其要覆盖 yellow_double / tractor_green，因为这两个是历史弱项。
4. 如果 INT8 掉点明显，比赛阶段可以优先使用 FP16/FP 模型。
```

LPRNet RKNN 输出必须保持：

```text
output shape = (1, 74, 18)
```

后处理仍然使用 CPU：

```text
logits -> CTC greedy / constrained beam search -> plate_text
```

不要把 CTC decode 塞进 ONNX 或 RKNN。

## 13. LPRNet 这次到底改了什么算法

这一点必须讲清楚：当前 LPRNet 已经不是最原始的“蓝牌一个模型、绿牌一个模型”的方案，也不是只靠 greedy decode 的原始 LPRNet。

### 13.1 模型层面：统一字符表

当前使用统一字符表：

```text
31 省份简称
10 数字
24 字母，去掉 I/O
8 特殊字符：学/挂/港/澳/使/领/警/临
1 blank
```

因此：

```text
class_num = 74
blank_index = 73
output = (1, 74, 18)
```

后续代码里的 `CHARS` 必须和：

```text
LPRNet_Pytorch/data/load_data.py
```

完全一致，顺序不能变。

### 13.2 训练层面：统一模型识别多类车牌

当前主模型训练数据包含：

```text
blue
green
yellow
black
```

并通过文件名 / 目录记录 subtype：

```text
green_normal
tractor_green
yellow_single
yellow_double
black
blue
```

当前推荐的 `p15_focus` 模型，是在 `p15_flatten` 基础上进一步提高特殊弱项训练占比得到的版本：

```text
重点加强:
  tractor_green
  yellow_double
```

它是当前主模型，不是只识别蓝绿牌。

### 13.3 导出层面：新增 LPRNet_export.py

当前训练模型：

```text
LPRNet_Pytorch/model/LPRNet.py
```

原 forward 写法里有动态遍历和 forward 内动态创建 AvgPool：

```text
for i, layer in enumerate(self.backbone.children())
nn.AvgPool2d(...)(f)
```

这对 PyTorch 训练没问题，但对 ONNX/RKNN 不够稳定。

所以新增导出专用模型：

```text
LPRNet_Pytorch/model/LPRNet_export.py
```

改动思想：

```text
1. 保持网络结构、参数名和训练模型一致。
2. 把 forward 改成静态顺序。
3. 显式定义 AvgPool 层。
4. 导出前加载同一个 .pth，验证训练版和导出版输出完全一致。
```

已经验证：

```text
PyTorch train/export max diff: 0.0000000000
```

这说明导出版模型没有改变识别算法，只是把计算图写得更适合 ONNX/RKNN。

### 13.4 解码层面：constrained CTC beam search

当前识别后处理不只使用 greedy decode。推荐使用：

```text
constrained_ctc_decode_one
```

位置：

```text
LPRNet_Pytorch/test_LPRNet.py
```

它做了两类增强：

```text
1. CTC beam search:
   保留多个候选路径，避免 greedy 每个时间步只取最大值导致局部错误。

2. 格式约束:
   根据 plate_type / plate_subtype 限制长度和字符位置。
```

当前长度规则：

```text
blue/yellow/black/unknown_7/special_7: 7
green/unknown_8: 8
```

当前字符规则：

```text
普通车牌:
  第 1 位省份
  第 2 位字母
  后续字母或数字

yellow:
  后续允许 学/挂

tractor_green:
  后续允许部分特殊字符

special_7:
  后续允许 学/挂/港/澳/使/领/警/临

black:
  当前规则较宽，允许非 blank 字符
```

这就是文档里说的“LPR 改了算法”的核心：模型输出 logits 不变，但解码从普通 greedy 升级成了“CTC beam search + 车牌格式动态约束”。

### 13.5 板端必须保留这个后处理策略

如果后续板端只实现 greedy decode，也能跑出结果，但精度可能低于当前 Python 测试结果。

要尽量复现当前效果，板端应实现：

```text
1. greedy CTC decode：用于最小闭环和调试。
2. constrained beam search：用于最终识别。
3. plate_type / plate_subtype 规则：用于限制候选。
4. 不确定 plate_type 时，多类型尝试后选最高分合法结果。
```

## 14. 后续实现优先级

### 14.1 第一阶段：只验证 RKNN 模型前向

目标：

```text
确认 YOLO RKNN 和 LPRNet RKNN 都能在板端跑起来。
```

要做：

```text
1. YOLO ONNX -> RKNN，先 fp。
2. LPRNet ONNX -> RKNN，先 fp。
3. 打印 YOLO 9 个 output 的 shape。
4. 打印 LPRNet output shape，确认是 (1,74,18)。
```

不做：

```text
不做透视矫正。
不先做 INT8。
不先写复杂业务逻辑。
```

### 14.2 第二阶段：YOLO 后处理

目标：

```text
从 YOLO 9 outputs 得到原图坐标下的 plate/person/car/traffic_light 检测框。
```

要做：

```text
1. letterbox 记录 scale / pad。
2. DFL decode。
3. conf threshold。
4. class-wise NMS。
5. scale back 到原图。
6. 保存可视化结果，确认框位置正确。
```

验收：

```text
同一张图上，RKNN 后处理结果和 Python/ONNX 后处理结果基本一致。
```

### 14.3 第三阶段：LPRNet 后处理

目标：

```text
从 LPRNet logits=(1,74,18) 得到车牌文本。
```

要做：

```text
1. crop resize 94x24。
2. 归一化保持 (img - 127.5) / 128。
3. greedy CTC decode。
4. constrained CTC beam search。
5. CHARS index 转 UTF-8 字符串。
```

验收：

```text
同一个 crop 上，RKNN 输出解码结果和 Python ONNX / PyTorch 结果一致或接近。
```

### 14.4 第四阶段：YOLO + LPR 串联

目标：

```text
整图输入，输出每个车牌的 bbox + plate_text。
```

要做：

```text
1. 只把 class_id=0 的 plate 框送入 LPRNet。
2. crop 加少量 padding。
3. HSV 粗判 plate_type。
4. plate_type 不确定时，多类型 constrained decode。
5. 多车牌按 y/x 排序输出。
```

输出格式建议：

```json
{
  "bbox": [x1, y1, x2, y2],
  "det_score": 0.91,
  "plate_type": "green",
  "plate_text": "粤BD12345",
  "lpr_score": 0.86,
  "decode_mode": "constrained"
}
```

### 14.5 第五阶段：量化和性能优化

目标：

```text
在保证精度的前提下提升板端速度。
```

顺序：

```text
1. FP RKNN 跑通。
2. YOLO 尝试 INT8。
3. LPRNet 尝试 INT8。
4. 对比 p15_focus 和 p2_province_focus。
5. 如果 LPRNet INT8 掉点明显，保留 FP/FP16 LPRNet。
```

不要一开始就直接 INT8，因为如果结果不对，很难判断问题来自：

```text
模型转换
量化
前处理
后处理
CHARS 映射
坐标裁剪
```

## 15. 最终交给后续 AI 的任务描述

可以直接把下面这段交给后续 AI：

```text
请基于当前项目实现瑞芯微 RK3588 上的 YOLO + LPRNet 串联部署。

当前已有:
1. YOLO RKNN 友好 ONNX:
   runs/yolo4_lite_n_50e_local_b8_amp_v2/weights/best.onnx
   注意同时需要 best.onnx.data。
   输出为 9 outputs，无内置 NMS。

2. LPRNet 主模型 ONNX:
   LPRNet_Pytorch/weights/unified_p15_focus/lprnet_unified_p15_focus_sim.onnx
   输入 (1,3,24,94)，输出 logits=(1,74,18)。

3. LPRNet 备选 ONNX:
   LPRNet_Pytorch/weights/unified_p2_province_focus/lprnet_unified_p2_province_focus_sim.onnx

4. LPRNet 当前不是原始 greedy-only 方案。
   它使用统一 CHARS=74，并推荐 constrained CTC beam search。
   参考 LPRNet_Pytorch/test_LPRNet.py。

请完成:
1. 使用 rknn_model_zoo 或自写 convert 脚本，把 YOLO / LPRNet ONNX 转 RKNN。
2. 先转 fp 版本跑通，再考虑 i8/u8 量化。
3. YOLO 后处理实现 DFL decode、阈值过滤、按类别 NMS、letterbox 坐标还原。
4. 只将 class_id=0 的 plate 检测框 crop 后送入 LPRNet。
5. LPRNet 后处理实现 greedy CTC decode 和 constrained beam search decode。
6. CHARS 顺序必须和 LPRNet_Pytorch/data/load_data.py 完全一致，blank index=73。
7. plate_type 先通过 crop HSV 粗判；不确定时用多种 plate_type 约束分别 decode，取最高分合法候选。
8. 当前版本不做透视矫正，不需要角点检测或四点变换。
9. 输出每个车牌的 bbox、det_score、plate_type、plate_text、lpr_score。
10. 代码结构需要方便以后迁移到 C++ RKNN C API。
```
## 16. rknn_model_zoo LPRNet 示例与当前模型的关系

当前已经补充专门说明：

```text
rknn_model_zoo-2.3.2/examples/LPRNet/README_当前项目使用说明.md
```

核心结论：

```text
1. 不使用官方 examples/LPRNet/python/export_onnx.py 导出当前模型。
2. 官方 export_onnx.py 写死 class_num=68，并且会使用官方旧权重。
3. 当前项目 LPRNet 是 class_num=74，必须使用：
   LPRNet_Pytorch/export_unified_lprnet_onnx.py
4. 已修改官方 examples/LPRNet/python/convert.py，可直接把当前 ONNX 转 RKNN。
5. 官方 Python/C++ demo 的 CHARS 和 greedy decode 不能直接用于当前最终模型。
6. 当前最终后处理参考：
   LPRNet_Pytorch/test_LPRNet.py
   constrained_ctc_decode_one(...)
```

当前 `convert.py` 已适配：

```text
1. 默认 mean_values = [127.5, 127.5, 127.5]
2. 默认 std_values = [128.0, 128.0, 128.0]，与训练预处理 (img - 127.5) / 128 对齐
3. 支持 --dataset 指定量化校准 txt，不需要再复制成 ../model/dataset.txt
4. 支持 --mean / --std / --verbose
```

推荐转换输入：

```text
LPRNet_Pytorch/weights/unified_p15_focus/lprnet_unified_p15_focus_sim.onnx
```

如果转 INT8，量化清单使用：

```text
/home/takigawayuki/PC_ubuntu22.04_26ICProject/Quantification_data/export_quant_data/lpr_quant_images.txt
```

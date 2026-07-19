## 七、板端 Python 推理（视频流）



环境装好后，先用 **Python** 把视频流跑通——它无需编译、改几行参数就能出结果，最适合快速验证模型在板端到底好不好使。（追求极致帧率的 C++ 多线程版本放在第八节。）

### 1. 先认识 RK3588 的 3 个 NPU 核



**RK3588 内置 3 个 NPU 核**（合计算力 6 TOPS）。`rknnlite` 在 `init_runtime(core_mask=...)` 里决定模型跑在哪个/哪些核上，常用取值：

| `core_mask`               | 含义                                        |
| ------------------------- | ------------------------------------------- |
| `RKNNLite.NPU_CORE_0`     | 只用 0 号核（单路视频流默认够用）           |
| `RKNNLite.NPU_CORE_0_1`   | 0、1 两核协同跑同一个模型                   |
| `RKNNLite.NPU_CORE_0_1_2` | 三核全开协同                                |
| `RKNNLite.NPU_CORE_AUTO`  | 自动调度：多个推理 context 自动分配到空闲核 |

**单路视频检测怎么选？** 实测中 YOLOv8n 这种小模型，**NPU 推理本身完全不是瓶颈**——在 RK3588 上单核纯推理就有约 50 FPS，三核（`NPU_CORE_0_1_2`）可达约 70 FPS。真正卡帧率的是 **CPU 侧的后处理**：如果照搬官方示例对全部 8400 个 anchor 都做一遍 DFL softmax，几十毫秒就耗在这；而**先用类别阈值掩码筛掉绝大多数 anchor、只对存活的几十个做 DFL**，后处理就几乎不耗时。本节脚本默认用三核 + 这个后处理优化，单路视频实测 **约 40 FPS（含画框）**。想进一步靠「多线程 + 绑核」榨干三核的，放在第八节 C++ 多线程里展开。

> ⚠️ **一个关键坑**：airockchip 导出的这个模型，`**cls` 分支输出已经是 sigmoid 后的概率**（数值范围 0~1），后处理里**千万不要再 sigmoid 一次**——否则背景 anchor（概率 0.3、0.4 那些）全部越过阈值，存活 anchor 从几十个暴涨到几千个，DFL/NMS 直接拖垮，帧率掉到个位数。

### 2. 完整推理脚本



下面是一份**自包含**的视频推理脚本（纯 `numpy` 后处理，板端无需安装 `torch`），匹配我们 9 输出的 3 类模型。建议存为 `infer_video.py`，放在 `rknn_model_zoo-main/examples/yolov8/python/` 下。

> **只需改顶部「配置区」这几行**（模型 / 视频 / 输出路径、类别名、阈值），其余后处理逻辑无需改动：

```
import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

# ========== 配置区：只改这几行 ==========
RKNN_MODEL   = '/mnt/RKNN_Garbage_Detection/basketball_rknn/basketball_yolov8_3cls_int8.rknn'  # ← 你的 rknn 模型
VIDEO_PATH   = '/mnt/RKNN_Garbage_Detection/basketball_rknn/basketball_kuli.mp4'                # ← 输入视频
OUTPUT_VIDEO = '/mnt/RKNN_Garbage_Detection/basketball_rknn/result.mp4'                         # ← 结果保存路径
CLASSES      = ("ball", "human", "rim")   # ← 类别名，顺序严格对应训练时的 data.yaml
OBJ_THRESH   = 0.25                        # 置信度阈值
NMS_THRESH   = 0.45                        # NMS IoU 阈值
IMG_SIZE     = (640, 640)                  # 模型输入尺寸 (w, h)
# =======================================

# 预计算 3 个尺度的 grid 与 DFL 投影向量（避免每帧重复构造，省 CPU）
STRIDES = (8, 16, 32)
GRIDS = {}
for s in STRIDES:
    gh, gw = IMG_SIZE[1] // s, IMG_SIZE[0] // s
    xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
    GRIDS[s] = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)
DFL_PROJ = np.arange(16, dtype=np.float32)


def post_process(outputs):
    """先用类别阈值掩码筛掉绝大多数 anchor，只对存活的 anchor 做 DFL 解码——
    这是把后处理从“全量 8400 anchor”降到“几十个 anchor”的关键提速点。
    注意：airockchip 导出的 cls 分支输出已是 sigmoid 后的概率，无需再 sigmoid。"""
    boxes_all, scores_all, classes_all = [], [], []
    for i, stride in enumerate(STRIDES):
        box_feat = outputs[i * 3]          # [1,64,h,w]  DFL 编码的 ltrb
        cls_feat = outputs[i * 3 + 1]      # [1, 3,h,w]  3 类概率（已 sigmoid）

        cls = cls_feat.reshape(cls_feat.shape[1], -1).T   # [h*w, 3]
        cls_max = cls.max(axis=1)
        mask = cls_max >= OBJ_THRESH
        if not np.any(mask):
            continue

        box = box_feat.reshape(64, -1).T[mask]      # [N,64]
        grid = GRIDS[stride][mask]                  # [N,2]
        scores = cls_max[mask]
        classes = np.argmax(cls[mask], axis=1)

        # DFL 解码：64 = 4 边 × 16 bin，softmax 后求期望
        b = box.reshape(-1, 4, 16)
        b = b - b.max(axis=2, keepdims=True)
        b = np.exp(b)
        b = b / b.sum(axis=2, keepdims=True)
        b = (b * DFL_PROJ).sum(axis=2)              # [N,4] = ltrb 距离

        x1 = (grid[:, 0] + 0.5 - b[:, 0]) * stride
        y1 = (grid[:, 1] + 0.5 - b[:, 1]) * stride
        x2 = (grid[:, 0] + 0.5 + b[:, 2]) * stride
        y2 = (grid[:, 1] + 0.5 + b[:, 3]) * stride

        boxes_all.append(np.stack([x1, y1, x2, y2], axis=-1))
        scores_all.append(scores)
        classes_all.append(classes)

    if not boxes_all:
        return None, None, None
    boxes = np.concatenate(boxes_all)
    scores = np.concatenate(scores_all)
    classes = np.concatenate(classes_all)

    # OpenCV 内置 NMS（C 实现，比纯 numpy 循环快），输入需 xywh
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    rects = np.concatenate([boxes[:, 0:2], wh], axis=1).tolist()
    keep = cv2.dnn.NMSBoxes(rects, scores.tolist(), OBJ_THRESH, NMS_THRESH)
    if len(keep) == 0:
        return None, None, None
    keep = np.array(keep).flatten()
    return boxes[keep], classes[keep], scores[keep]


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def draw(image, boxes, scores, classes, ratio, pad):
    for box, score, cl in zip(boxes, scores, classes):
        x1 = int((box[0] - pad[0]) / ratio)
        y1 = int((box[1] - pad[1]) / ratio)
        x2 = int((box[2] - pad[0]) / ratio)
        y2 = int((box[3] - pad[1]) / ratio)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{CLASSES[cl]} {score:.2f}', (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image


if __name__ == '__main__':
    rknn = RKNNLite()
    print('--> Load RKNN model')
    assert rknn.load_rknn(RKNN_MODEL) == 0, 'Load RKNN model failed!'

    print('--> Init runtime')
    # RK3588 三核全开，单路视频也能压满 NPU；想省功耗可换成 NPU_CORE_0
    assert rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, 'Init runtime failed!'

    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f'Cannot open video: {VIDEO_PATH}'
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    print(f'Video: {w}x{h}, {fps}fps, {total} frames')

    frame_count, total_time = 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        t0 = time.time()

        img, ratio, pad = letterbox(frame, new_shape=IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)        # 模型输入 NHWC、UINT8，直接喂 0~255

        outputs = rknn.inference(inputs=[img])
        boxes, classes, scores = post_process(outputs)

        dt = time.time() - t0
        total_time += dt
        cur_fps = 1.0 / dt if dt > 0 else 0
        avg_fps = frame_count / total_time

        if boxes is not None:
            frame = draw(frame, boxes, scores, classes, ratio, pad)
        cv2.putText(frame, f'FPS: {cur_fps:.1f} (Avg: {avg_fps:.1f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out.write(frame)
        print(f'Frame {frame_count}/{total} | FPS: {cur_fps:.1f} | Avg: {avg_fps:.1f}', end='\r')

    print(f'\nDone! {frame_count} frames, Avg FPS: {avg_fps:.1f}')
    print(f'Result saved to: {OUTPUT_VIDEO}')
    cap.release(); out.release(); rknn.release()
```



### 3. 运行



在 `(basketball)` 环境下，直接执行脚本即可：

```
conda activate basketball
python infer_video.py
```



终端会逐帧打印当前/平均 FPS，跑完后在 `OUTPUT_VIDEO` 指定路径得到一段带检测框的结果视频。本例 720×1092 的篮球视频实测**平均 ~40 FPS**（左上角红字实时显示），三类 `ball / human / rim` 均能稳定框出：

[![Python 视频流推理结果（左上角实时 FPS，平均 ~40 FPS）](https://github.com/JA-cmd-wq/yolov8-helmet-rk3588-multithread/raw/main/docs/images/24-python-infer-result.png)](https://github.com/JA-cmd-wq/yolov8-helmet-rk3588-multithread/blob/main/docs/images/24-python-infer-result.png)

> 💡 **脚本里的 FPS 统计的是「前处理 + 推理 + 后处理」的计算耗时**（不含写 mp4 的编码时间），所以它反映的是模型在板端的真实处理能力。若把结果改成实时 `imshow` 显示而不写文件，帧率还会更高一些。

> **几个常见可调项：**
>
> - **换自己的任务**：只改配置区的 `RKNN_MODEL`、`VIDEO_PATH`、`CLASSES`（顺序务必对应 `data.yaml` 的 `names`）。
> - **实时预览**：若板子接了显示器，可在 `out.write(frame)` 后加 `cv2.imshow('result', frame); cv2.waitKey(1)`。
> - **跑摄像头**：把 `VIDEO_PATH` 换成摄像头索引，即 `cv2.VideoCapture(0)`。
> - **掉框/误检多**：适当调高 `OBJ_THRESH`；漏检多则调低。










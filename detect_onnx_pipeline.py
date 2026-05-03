import cv2
import numpy as np
import onnxruntime as ort
import os
from PIL import Image, ImageDraw, ImageFont
from LPRNet_Pytorch.data.load_data import CHARS

# 中文字体路径（Windows自带）
_FONT = None
def _get_font(size=20):
    global _FONT
    if _FONT is None:
        for path in [
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
        ]:
            if os.path.exists(path):
                _FONT = ImageFont.truetype(path, size)
                break
        if _FONT is None:
            _FONT = ImageFont.load_default()
    return _FONT


def put_chinese_text(img_bgr, text, pos, color=(0, 255, 0), size=20):
    """在 BGR 图上用 PIL 画支持中文的文字"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=_get_font(size), fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# =========================
# 1. 模型路径
# =========================
YOLO_PATH = r"C:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project\runs\new_plate_detect_merged\weights\best.onnx"

LPR_BLUE_PATH = r"C:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project\LPRNet_Pytorch\weights\blue_re_run1\blue_LPRNet_Simplified.onnx"

LPR_GREEN_PATH = r"C:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project\LPRNet_Pytorch\weights\green_re_run1\green_LPRNet_Simplified.onnx"


# =========================
# 2. ONNX Session
# =========================
yolo_sess = ort.InferenceSession(YOLO_PATH, providers=["CPUExecutionProvider"])
lpr_blue = ort.InferenceSession(LPR_BLUE_PATH, providers=["CPUExecutionProvider"])
lpr_green = ort.InferenceSession(LPR_GREEN_PATH, providers=["CPUExecutionProvider"])

print("CHARS len:", len(CHARS))
print("CHARS head:", CHARS[:10])
print("CHARS tail:", CHARS[-10:])

# =========================
# 3. 工具函数
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def letterbox(img, new_size=640):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh))
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2

    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = img_resized

    return canvas, scale, left, top


# =========================
# 4. YOLO preprocess
# =========================
def preprocess_yolo(img):
    img, scale, left, top = letterbox(img, 640)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img, scale, left, top


# =========================
# 5. 简单IOU NMS
# =========================
def nms(boxes, scores, iou_thres=0.5):

    # 调试信息
    print("boxes shape:", boxes.shape)
    print("scores shape:", scores.shape)

    if len(boxes) == 0:
        return []

    if len(boxes) == 1:
        return [0]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


# =========================
# 6. YOLO inference（修复工业版）
# =========================
def scale_back(boxes, scale, left, top):
    """
    将 letterbox 640坐标 → 原图坐标
    """
    boxes = boxes.copy()

    boxes[:, 0] = (boxes[:, 0] - left) / scale
    boxes[:, 2] = (boxes[:, 2] - left) / scale
    boxes[:, 1] = (boxes[:, 1] - top) / scale
    boxes[:, 3] = (boxes[:, 3] - top) / scale

    return boxes


def decode_yolo(outputs, conf_thres=0.3):
    outputs = outputs[0]

    print("sample boxes:", outputs[:5])

    # N x 6
    if outputs.shape[-1] == 6:
        boxes = outputs[:, :4]
        scores = outputs[:, 4]
        cls = outputs[:, 5]

    elif outputs.shape[0] == 6:
        outputs = outputs.T
        boxes = outputs[:, :4]
        scores = outputs[:, 4]
        cls = outputs[:, 5]
    else:
        raise ValueError(f"Unknown shape: {outputs.shape}")

    # ⭐⭐⭐ 位置1：过滤非法框（就在这里）
    valid = (boxes[:, 0] > 0) & (boxes[:, 1] > 0)
    boxes = boxes[valid]
    scores = scores[valid]
    cls = cls[valid]

    # sigmoid
    scores = 1 / (1 + np.exp(-scores))

    mask = scores > conf_thres

    return boxes[mask], scores[mask], cls[mask].astype(int)


def yolo_infer(img):

    inp, scale, left, top = preprocess_yolo(img)

    outputs = yolo_sess.run(None, {"images": inp})[0]

    boxes, scores, classes = decode_yolo(outputs)

    print("raw boxes:", boxes.shape, "scores:", scores.shape)  # ✔改这里

    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    print("min/max x:", boxes[:,0].min(), boxes[:,0].max())
    print("min/max y:", boxes[:,1].min(), boxes[:,1].max())

    keep = nms(boxes, scores)

    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    boxes = scale_back(boxes, scale, left, top)

    print("FINAL boxes:", boxes)
    print("FINAL scores:", scores)
    print("FINAL classes:", classes)  # 0=blue, 1=green

    return boxes, scores, classes


# =========================
# 7. crop
# =========================
def crop(img, box):

    box = np.asarray(box).reshape(4,)

    x1, y1, x2, y2 = box.astype(int)

    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return img[y1:y2, x1:x2]


# =========================
# 8. 车牌颜色判断（简单版）
# =========================
def is_green_plate(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    ratio = np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])

    return ratio > 0.15


# =========================
# 8b. 车牌格式校验
# =========================
_PROVINCE = set(CHARS[:31])          # 京沪津渝…新（前31个汉字）
_LETTERS  = set('ABCDEFGHJKLMNPQRSTUVWXYZ')   # 无 I O
def plate_format_score(text):
    """
    返回格式分 0~3，越高越符合中国车牌规范：
      +1  长度合法（7位蓝牌 或 8位绿牌/新能源）
      +1  首字符是省份汉字
      +1  第二字符是字母
    """
    score = 0
    if len(text) in (7, 8):
        score += 1
    if len(text) >= 1 and text[0] in _PROVINCE:
        score += 1
    if len(text) >= 2 and text[1] in _LETTERS:
        score += 1
    return score


# =========================
# 9. LPR preprocess
# =========================
def lpr_preprocess(img):
    img = cv2.resize(img, (94, 24))
    img = img.astype(np.float32)
    img = (img - 127.5) * 0.0078125
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


# =========================
# 10. LPR inference
# =========================
def ctc_decode(pred, chars):

    # stable softmax
    pred = pred - np.max(pred, axis=0, keepdims=True)
    prob = np.exp(pred)
    prob = prob / np.sum(prob, axis=0, keepdims=True)
    
    raw = np.argmax(prob, axis=0)  # T维

    blank = len(chars) - 1

    # ===== debug 1 =====
    print("raw decode:", raw)
    print("max index:", np.max(raw), "chars len:", len(chars))

    # ===== debug 2（blank比例）=====
    print("blank ratio:", np.sum(raw == blank) / len(raw))

    # ===== debug 3（分布）=====
    print("unique raw:", np.unique(raw))
    print("blank index:", blank)

    # ===== debug 4（字符序列）=====
    print("char sequence:", [chars[i] for i in raw if i != blank])

    result = []
    conf_vals = []
    prev = None

    for t, c in enumerate(raw):

        if c == blank:
            prev = None   # ⭐关键修复
            continue

        if c != prev:
            result.append(chars[c])
            conf_vals.append(float(np.max(prob, axis=0)[t]))

        prev = c

    confidence = float(np.mean(conf_vals)) if conf_vals else 0.0
    return "".join(result), confidence


def lpr_infer(img, model, chars):
    inp = lpr_preprocess(img)

    print(lpr_blue.get_inputs()[0].name)
    print(lpr_green.get_inputs()[0].name)

    input_name = model.get_inputs()[0].name
    out = model.run(None, {input_name: inp})[0]

    out = out[0]  # (C, T)

    # ⭐⭐⭐ 加在这里
    print("raw model output shape:", model.run(None, {input_name: inp})[0].shape)
    print("after squeeze shape:", out.shape)

    # ⭐⭐⭐ 就加这一行
    print("std per timestep:", np.std(out, axis=0)[:10])

    print("argmax axis0:", np.argmax(out, axis=0)[:10])
    print("argmax axis1:", np.argmax(out, axis=1)[:10])

    print("argmax axis1 distribution:", np.bincount(np.argmax(out, axis=0).flatten())[:10])
    print("out min/max:", out.min(), out.max())
    print("out shape:", out.shape)

    result, confidence = ctc_decode(out, chars)

    return result, confidence


def lpr_infer_best(img, chars):
    """
    路由优先级：
    1. 长度信号：绿模型8位 且 蓝模型非8位 → 新能源绿牌，选绿
                  蓝模型7位 且 绿模型非7位 → 标准蓝牌，选蓝
    2. 长度相同或两者都不符合 → 格式分决胜
    3. 格式分相同 → 置信度决胜
    """
    text_b, conf_b = lpr_infer(img, lpr_blue, chars)
    text_g, conf_g = lpr_infer(img, lpr_green, chars)

    fmt_b = plate_format_score(text_b)
    fmt_g = plate_format_score(text_g)

    print(f"  blue:  {text_b!r}  len={len(text_b)}  fmt={fmt_b}  conf={conf_b:.3f}")
    print(f"  green: {text_g!r}  len={len(text_g)}  fmt={fmt_g}  conf={conf_g:.3f}")

    # Step 1: 蓝模型三项全对（省份+字母+7位）→ 一定是标准蓝牌
    if fmt_b == 3 and len(text_b) == 7:
        return text_b, conf_b, "BLUE"

    # Step 2: 蓝模型不完美，且绿模型输出8位 → 新能源绿牌
    if len(text_g) == 8:
        return text_g, conf_g, "GREEN"

    # Step 3: 绿模型也不是8位，蓝模型7位
    if len(text_b) == 7:
        return text_b, conf_b, "BLUE"

    # Step 4: 都不标准 → 格式分 → 置信度
    if fmt_g > fmt_b:
        return text_g, conf_g, "GREEN"
    if fmt_b > fmt_g:
        return text_b, conf_b, "BLUE"
    if conf_g >= conf_b:
        return text_g, conf_g, "GREEN"
    return text_b, conf_b, "BLUE"


# =========================
# 11. 主流程
# =========================
def main(img_path):
    # img = cv2.imread(img_path)

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    boxes, scores, classes = yolo_infer(img)

    for box, score, cls in zip(boxes, scores, classes):
        plate = crop(img, box)

        if plate.size == 0:
            continue

        result, lpr_conf, color = lpr_infer_best(plate, CHARS)

        print(f"[{color}(yolo_cls={cls})] {result}  det_conf={score:.2f}  lpr_conf={lpr_conf:.3f}")

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        img = put_chinese_text(img, result, (x1, max(0, y1-26)), color=(0, 255, 0), size=22)

    cv2.imshow("result", img)
    cv2.waitKey(0)


# =========================
# 12. run
# =========================
if __name__ == "__main__":
    test_img = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019\ccpd_base\045-92_82-219&352_589&493-601&494_208&484_214&353_607&363-0_0_9_32_31_32_30-87-94.jpg"  # 改成你的图片
    main(test_img)
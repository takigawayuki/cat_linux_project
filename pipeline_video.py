import os
import cv2
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append('/home/cat/cat_linux_project/rknn_model_zoo')
from py_utils.coco_utils import COCO_test_helper
from py_utils.rknn_executor import RKNN_model_container
from rknnlite.api import RKNNLite
from Drivers.camera import Camera

# ══════════════════════════════════════════════════════════════════════════════
# 在这里修改路径和参数
# ══════════════════════════════════════════════════════════════════════════════
BASE = '/home/cat/cat_linux_project'

YOLO_MODEL  = f'{BASE}/RKNN_NEW/yolov8n_notnms.rknn'
LPR_BLUE    = f'{BASE}/RKNN_NEW/blue_re_run1.rknn'
LPR_GREEN   = f'{BASE}/RKNN_NEW/green_re_run1.rknn'
TARGET      = 'rk3568'
# ══════════════════════════════════════════════════════════════════════════════

OBJ_THRESH  = 0.70
NMS_THRESH  = 0.35
IMG_SIZE    = (640, 640)
FONT_PATH   = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-']

_PROVINCE = set(CHARS[:31])
_LETTERS  = set('ABCDEFGHJKLMNPQRSTUVWXYZ')


def nms_boxes(boxes, scores):
    x = boxes[:, 0]; y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]; h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]]); yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i]+w[i], x[order[1:]]+w[order[1:]])
        yy2 = np.minimum(y[i]+h[i], y[order[1:]]+h[order[1:]])
        inter = np.maximum(0, xx2-xx1+1e-5) * np.maximum(0, yy2-yy1+1e-5)
        ovr   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= NMS_THRESH)[0] + 1]
    return np.array(keep)


def yolo_post_process(outputs):
    out = np.squeeze(outputs[0])   # (5, 8400)
    cx, cy, w, h, conf = out[0], out[1], out[2], out[3], out[4]
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    boxes  = np.stack([x1, y1, x2, y2], axis=1)
    mask   = conf > OBJ_THRESH
    boxes, scores = boxes[mask], conf[mask]
    if len(boxes) == 0:
        return None, None
    keep = nms_boxes(boxes, scores)
    return boxes[keep], scores[keep]


def lpr_decode(preds):
    blank  = len(CHARS) - 1
    logits = preds[0]
    ls     = logits - np.max(logits, axis=0, keepdims=True)
    probs  = np.exp(ls) / np.sum(np.exp(ls), axis=0, keepdims=True)
    seq    = np.argmax(logits, axis=0)
    result = []; pre = blank
    for c in seq:
        if c == blank: pre = blank; continue
        if c != pre: result.append(CHARS[c])
        pre = c
    text = ''.join(result)
    conf = float(np.mean(np.max(probs, axis=0)[seq != blank])) if np.any(seq != blank) else 0.0
    return text, conf


def plate_format_score(text):
    s = 0
    if len(text) in (7, 8): s += 1
    if len(text) >= 1 and text[0] in _PROVINCE: s += 1
    if len(text) >= 2 and text[1] in _LETTERS: s += 1
    return s


def lpr_infer_best(crop_bgr, lpr_blue, lpr_green):
    inp   = cv2.resize(crop_bgr, (94, 24))[np.newaxis, :]
    out_b = lpr_blue.inference(inputs=[inp])
    out_g = lpr_green.inference(inputs=[inp])
    tb, cb = lpr_decode(out_b[0])
    tg, cg = lpr_decode(out_g[0])
    sb = cb + plate_format_score(tb) * 0.1
    sg = cg + plate_format_score(tg) * 0.1
    return (tg, cg, 'GREEN') if sg >= sb else (tb, cb, 'BLUE')


def draw_result(img_bgr, x1, y1, x2, y2, plate_text, color_tag, score):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(FONT_PATH, size=28)
    except Exception:
        font = ImageFont.load_default()
    box_color = (0, 220, 0) if color_tag == 'BLUE' else (0, 200, 100)
    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
    label = f'[{color_tag}] {plate_text}  {score:.2f}'
    ty    = max(y1 - 34, 0)
    draw.text((x1, ty), label, font=font, fill=box_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    print('--> Loading models')
    yolo = RKNN_model_container(YOLO_MODEL, TARGET, None)
    lpr_blue  = RKNNLite(verbose=False)
    lpr_blue.load_rknn(LPR_BLUE); lpr_blue.init_runtime()
    lpr_green = RKNNLite(verbose=False)
    lpr_green.load_rknn(LPR_GREEN); lpr_green.init_runtime()
    print('Models loaded')

    co = COCO_test_helper(enable_letter_box=True)

    win = 'License Plate Recognition'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    with Camera(device=9, width=640, height=480, fps=60) as cam:
        print('Press q or ESC to quit')
        fps_display = 0.0
        t_last = time.time()

        while True:
            frame = cam.read()

            img_lb = co.letter_box(im=frame.copy(),
                                   new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
                                   pad_color=(0, 0, 0))
            img_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            yolo_out = yolo.run([img_lb[np.newaxis, :]])
            boxes, scores = yolo_post_process(yolo_out)

            vis = frame.copy()

            if boxes is not None:
                real_boxes = co.get_real_box(boxes)
                for box, score in zip(real_boxes, scores):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    plate, lpr_conf, color_tag = lpr_infer_best(crop, lpr_blue, lpr_green)
                    print(f'[{color_tag}] {plate}  det={score:.3f}  lpr={lpr_conf:.3f}')
                    vis = draw_result(vis, x1, y1, x2, y2, plate, color_tag, score)

            # 计算并平滑显示帧率
            t_now = time.time()
            fps_instant = 1.0 / max(t_now - t_last, 1e-6)
            fps_display = fps_display * 0.9 + fps_instant * 0.1
            t_last = t_now

            # 左上角叠加帧率信息
            cam_fps = cam.cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(vis, f'CAM {cam_fps:.0f}fps  INF {fps_display:.1f}fps',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    yolo.release()
    lpr_blue.release()
    lpr_green.release()


if __name__ == '__main__':
    main()

import os
import cv2
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append('/home/cat/cat_linux_project/rknn_model_zoo')
from py_utils.coco_utils import COCO_test_helper
from py_utils.rknn_executor import RKNN_model_container
from rknnlite.api import RKNNLite

# ══════════════════════════════════════════════════════════════════════════════
# 在这里修改路径和参数，然后直接 python3 pipeline.py 运行
# ══════════════════════════════════════════════════════════════════════════════
BASE = '/home/cat/cat_linux_project'

DEFAULT_YOLO_MODEL = f'{BASE}/RKNN_NEW/yolov8n_notnms.rknn'
DEFAULT_LPR_BLUE   = f'{BASE}/RKNN_NEW/blue_re_run1.rknn'
DEFAULT_LPR_GREEN  = f'{BASE}/RKNN_NEW/green_re_run1.rknn'
DEFAULT_TARGET     = 'rk3568'

# 单张图片路径（设为 None 则用 DEFAULT_IMG_FOLDER）
DEFAULT_IMG_PATH   = "test_photo/Random/035-3_5-220&504_481&616-478&598_220&616_223&522_481&504-0_0_0_32_0_27_27-155-124.jpg"
# 文件夹路径（DEFAULT_IMG_PATH 不为 None 时忽略）
DEFAULT_IMG_FOLDER = f'{BASE}/test_photo/CCPD2019'

DEFAULT_IMG_SHOW   = True   # 弹出窗口显示
DEFAULT_IMG_SAVE   = False  # 保存到 result/
# ══════════════════════════════════════════════════════════════════════════════

# ── YOLO params ───────────────────────────────────────────────────────────────
OBJ_THRESH = 0.70
NMS_THRESH  = 0.35
IMG_SIZE    = (640, 640)

# ── LPRNet params ─────────────────────────────────────────────────────────────
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-']  # '-' at index 65 = CTC blank

_PROVINCE = set(CHARS[:31])
_LETTERS  = set('ABCDEFGHJKLMNPQRSTUVWXYZ')

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


# ── YOLO helpers ──────────────────────────────────────────────────────────────
def nms_boxes(boxes, scores):
    x = boxes[:, 0];  y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0];  h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0];  keep.append(i)
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
    x1 = cx - w/2;  y1 = cy - h/2;  x2 = cx + w/2;  y2 = cy + h/2
    boxes  = np.stack([x1, y1, x2, y2], axis=1)
    mask   = conf > OBJ_THRESH
    boxes, scores = boxes[mask], conf[mask]
    if len(boxes) == 0:
        return None, None
    keep = nms_boxes(boxes, scores)
    return boxes[keep], scores[keep]


# ── LPRNet helpers ────────────────────────────────────────────────────────────
def lpr_preprocess(crop_bgr):
    img = cv2.resize(crop_bgr, (94, 24))
    return img[np.newaxis, :]   # (1, 24, 94, 3), uint8


def lpr_decode(preds):
    """CTC greedy decode. preds shape: (1, 66, 18)"""
    blank  = len(CHARS) - 1
    logits = preds[0]  # (66, 18)
    # softmax for confidence
    logits_shifted = logits - np.max(logits, axis=0, keepdims=True)
    probs  = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=0, keepdims=True)
    seq    = np.argmax(logits, axis=0)
    result = []
    pre    = blank
    for c in seq:
        if c == blank:
            pre = blank
            continue
        if c != pre:
            result.append(CHARS[c])
        pre = c
    text = ''.join(result)
    conf = float(np.mean(np.max(probs, axis=0)[seq != blank])) if np.any(seq != blank) else 0.0
    return text, conf


def plate_format_score(text):
    """0~3: higher = more like a valid Chinese plate."""
    score = 0
    if len(text) in (7, 8):
        score += 1
    if len(text) >= 1 and text[0] in _PROVINCE:
        score += 1
    if len(text) >= 2 and text[1] in _LETTERS:
        score += 1
    return score


def lpr_infer_best(crop_bgr, lpr_blue, lpr_green):
    inp = lpr_preprocess(crop_bgr)

    out_b = lpr_blue.inference(inputs=[inp])
    out_g = lpr_green.inference(inputs=[inp])

    text_b, conf_b = lpr_decode(out_b[0])
    text_g, conf_g = lpr_decode(out_g[0])

    fmt_b = plate_format_score(text_b)
    fmt_g = plate_format_score(text_g)

    # weighted score: confidence + format bonus
    score_b = conf_b + fmt_b * 0.1
    score_g = conf_g + fmt_g * 0.1

    if score_g >= score_b:
        return text_g, conf_g, 'GREEN'
    return text_b, conf_b, 'BLUE'


# ── drawing ───────────────────────────────────────────────────────────────────
def draw_result(img_bgr, x1, y1, x2, y2, plate_text, color_tag, score):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(FONT_PATH, size=28)
    except Exception:
        font = ImageFont.load_default()

    box_color = (0, 200, 0) if color_tag == 'BLUE' else (0, 180, 80)
    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
    label = f'[{color_tag}] {plate_text}  {score:.2f}'
    ty    = max(y1 - 34, 0)
    draw.text((x1, ty), label, font=font, fill=box_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── main ──────────────────────────────────────────────────────────────────────
def img_check(p):
    return any(p.endswith(e) or p.endswith(e.upper())
               for e in ['.jpg', '.jpeg', '.png', '.bmp'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model',  type=str, default=None)
    parser.add_argument('--lpr_blue',    type=str, default=None)
    parser.add_argument('--lpr_green',   type=str, default=None)
    parser.add_argument('--target',      type=str, default=None)
    parser.add_argument('--img_folder',  type=str, default=None)
    parser.add_argument('--img_path',    type=str, default=None)
    parser.add_argument('--img_save',    action='store_true', default=None)
    parser.add_argument('--img_show',    action='store_true', default=None)
    args = parser.parse_args()

    yolo_model = args.yolo_model or DEFAULT_YOLO_MODEL
    lpr_blue_path  = args.lpr_blue   or DEFAULT_LPR_BLUE
    lpr_green_path = args.lpr_green  or DEFAULT_LPR_GREEN
    target     = args.target     or DEFAULT_TARGET
    img_path   = args.img_path   or DEFAULT_IMG_PATH
    img_folder = args.img_folder or (None if img_path else DEFAULT_IMG_FOLDER)
    img_show   = args.img_show   or DEFAULT_IMG_SHOW
    img_save   = args.img_save   or DEFAULT_IMG_SAVE

    if img_folder is None and img_path is None:
        print('Set DEFAULT_IMG_PATH or DEFAULT_IMG_FOLDER at the top of pipeline.py'); exit(1)

    print('--> Loading YOLO model')
    yolo = RKNN_model_container(yolo_model, target, None)
    print('done')

    print('--> Loading LPRNet blue model')
    lpr_blue = RKNNLite(verbose=False)
    lpr_blue.load_rknn(lpr_blue_path)
    lpr_blue.init_runtime()
    print('done')

    print('--> Loading LPRNet green model')
    lpr_green = RKNNLite(verbose=False)
    lpr_green.load_rknn(lpr_green_path)
    lpr_green.init_runtime()
    print('done')

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    if img_save:
        os.makedirs(result_dir, exist_ok=True)

    co_helper = COCO_test_helper(enable_letter_box=True)

    if img_path:
        file_list = [img_path]
    else:
        file_list = [os.path.join(img_folder, f)
                     for f in sorted(os.listdir(img_folder)) if img_check(f)]

    for fpath in file_list:
        img_src = cv2.imread(fpath)
        if img_src is None:
            continue
        fname = os.path.basename(fpath)

        # YOLO
        img_lb = co_helper.letter_box(im=img_src.copy(),
                                      new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
                                      pad_color=(0, 0, 0))
        img_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        yolo_out = yolo.run([img_lb[np.newaxis, :]])
        boxes, scores = yolo_post_process(yolo_out)

        if boxes is None:
            print(f'[no detection] {fname}')
            continue

        real_boxes = co_helper.get_real_box(boxes)
        img_draw   = img_src.copy()

        for box, score in zip(real_boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, img_src.shape[1]), min(y2, img_src.shape[0])

            crop = img_src[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            plate, lpr_conf, color_tag = lpr_infer_best(crop, lpr_blue, lpr_green)
            print(f'{fname[:50]}  [{color_tag}] {plate}  det={score:.3f}  lpr={lpr_conf:.3f}')

            if img_save or img_show:
                img_draw = draw_result(img_draw, x1, y1, x2, y2, plate, color_tag, score)

        if img_save:
            cv2.imwrite(os.path.join(result_dir, fname), img_draw)

        if img_show:
            cv2.imshow(fname, img_draw)
            while True:
                key = cv2.waitKey(100) & 0xFF
                # exit on 'q', ESC, or window closed
                if key in (ord('q'), 27):
                    break
                if cv2.getWindowProperty(fname, cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()

    yolo.release()
    lpr_blue.release()
    lpr_green.release()

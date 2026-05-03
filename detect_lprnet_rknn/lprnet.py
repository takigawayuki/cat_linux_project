import os
import sys
import cv2
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
from rknnlite.api import RKNNLite

# 66 classes matching model output dim=66; index 65 is CTC blank
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-']  # '-' at index 65 = CTC blank

# CCPD province table (index matches filename field 4, first char)
CCPD_PROVINCES = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                  '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                  '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                  '新', '警', '学', 'O']
# CCPD alphanumeric table (index matches filename field 4, chars 2-7)
CCPD_ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                  'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                  'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
                  '6', '7', '8', '9', 'O']


def decode(preds, CHARS):
    # CTC greedy decode; blank token is the last index
    blank = len(CHARS) - 1
    pred_labels = []
    labels = []
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = [np.argmax(pred[:, j], axis=0) for j in range(pred.shape[1])]
        no_repeat_blank = []
        pre_c = blank  # init to blank so first real char is never skipped
        for c in pred_label:
            if c == blank:
                pre_c = blank
                continue
            if c != pre_c:
                no_repeat_blank.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank)

    for label in pred_labels:
        labels.append(''.join(CHARS[i] for i in label))
    return labels, pred_labels


def parse_ccpd_filename(filename):
    """Parse bbox and plate label from CCPD filename."""
    name = os.path.splitext(filename)[0]
    parts = name.split('-')
    # field 2: bbox "x1&y1_x2&y2"
    bbox_str = parts[2].split('_')
    x1, y1 = [int(v) for v in bbox_str[0].split('&')]
    x2, y2 = [int(v) for v in bbox_str[1].split('&')]
    # field 4: plate label indices — first index uses CCPD_PROVINCES, rest use CCPD_ALPHABETS
    label_indices = [int(v) for v in parts[4].split('_')]
    gt_label = CCPD_PROVINCES[label_indices[0]] + ''.join(CCPD_ALPHABETS[i] for i in label_indices[1:])
    return (x1, y1, x2, y2), gt_label


def img_check(path):
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        if path.endswith(ext) or path.endswith(ext.upper()):
            return True
    return False


FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def draw_result(img_bgr, x1, y1, x2, y2, pred, gt):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    color = (0, 255, 0) if pred == gt else (255, 0, 0)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    try:
        font = ImageFont.truetype(FONT_PATH, size=28)
    except Exception:
        font = ImageFont.load_default()

    match = '✓' if pred == gt else '✗'
    top_text = f'pred: {pred}'
    bot_text = f'gt: {gt} {match}'

    # draw text above box (pred) and below box (gt)
    ty = max(y1 - 32, 0)
    draw.text((x1, ty), top_text, font=font, fill=color)
    by = min(y2 + 4, pil_img.height - 32)
    draw.text((x1, by), bot_text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPRNet Python Demo')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--target', type=str, default='rk3566')
    parser.add_argument('--device_id', type=str, default=None)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--result_file', type=str, default=None)
    parser.add_argument('--img_save', action='store_true')
    args = parser.parse_args()

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    if args.img_save:
        os.makedirs(result_dir, exist_ok=True)

    rknn = RKNNLite(verbose=False)

    ret = rknn.load_rknn(args.model_path)
    if ret != 0:
        print('Load RKNN model failed!')
        exit(ret)

    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    file_list = sorted([f for f in os.listdir(args.img_folder) if img_check(f)])
    correct = 0
    total = 0

    for fname in file_list:
        img_path = os.path.join(args.img_folder, fname)
        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        (x1, y1, x2, y2), gt_label = parse_ccpd_filename(fname)

        # crop and resize to LPRNet input size
        crop = img_src[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (94, 24))

        outputs = rknn.inference(inputs=[crop[np.newaxis, :]])

        labels, _ = decode(outputs[0], CHARS)
        pred = labels[0] if labels else ''

        match = '✓' if pred == gt_label else '✗'
        print(f'{match} pred: {pred:12s}  gt: {gt_label}  [{fname[:30]}...]')

        if args.img_save:
            out_img = draw_result(img_src, x1, y1, x2, y2, pred, gt_label)
            cv2.imwrite(os.path.join(result_dir, fname), out_img)

        if pred == gt_label:
            correct += 1
        total += 1

    print(f'\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%' if total > 0 else 'No images found')

    rknn.release()

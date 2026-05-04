"""
蓝牌 pipeline：FPGA UDP → ROI裁剪 → LPRNet识别
FPGA已做定位，ROI坐标由第241号包携带，直接用来裁剪车牌区域。
"""
import sys
import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from PIL import Image, ImageDraw, ImageFont

sys.path.append('/home/cat/cat_linux_project')
from Drivers.udp_camera import UDPCamera, WIDTH, HEIGHT

# ══════════════════════════════════════════════════════════════════════════════
BASE      = '/home/cat/cat_linux_project'
LPR_BLUE  = f'{BASE}/RKNN_NEW/blue_re_run1.rknn'
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
# ══════════════════════════════════════════════════════════════════════════════

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-']


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


def draw_result(img_bgr, x1, y1, x2, y2, plate_text, conf):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(FONT_PATH, size=18)
    except Exception:
        font = ImageFont.load_default()
    color = (0, 220, 0)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    draw.text((x1, max(y1 - 22, 0)), f'{plate_text} {conf:.2f}', font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    print('--> Loading LPRNet blue model')
    lpr = RKNNLite(verbose=False)
    lpr.load_rknn(LPR_BLUE)
    lpr.init_runtime()
    print('done')

    win = 'Blue Plate LPR (FPGA ROI)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    inf_fps = 0.0
    t_last  = time.time()

    print('Press q or ESC to quit')
    with UDPCamera() as cam:
        while True:
            frame, (x1, y1, x2, y2), cap_fps = cam.read()

            vis = frame.copy()

            xmin = max(min(x1, x2), 0); xmax = min(max(x1, x2), WIDTH)
            ymin = max(min(y1, y2), 0); ymax = min(max(y1, y2), HEIGHT)

            if xmax > xmin and ymax > ymin:
                crop = frame[ymin:ymax, xmin:xmax]
                if crop.size > 0:
                    inp = cv2.resize(crop, (94, 24))[np.newaxis, :]
                    out = lpr.inference(inputs=[inp])
                    plate, lpr_conf = lpr_decode(out[0])
                    print(f'[BLUE] {plate}  lpr={lpr_conf:.3f}')
                    vis = draw_result(vis, xmin, ymin, xmax, ymax, plate, lpr_conf)

            t_now   = time.time()
            inf_fps = inf_fps * 0.9 + (1.0 / max(t_now - t_last, 1e-6)) * 0.1
            t_last  = t_now
            cv2.putText(vis, f'CAP {cap_fps:.1f}fps  INF {inf_fps:.1f}fps',
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, f'CAP {cap_fps:.1f}fps  INF {inf_fps:.1f}fps',
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    lpr.release()


if __name__ == '__main__':
    main()

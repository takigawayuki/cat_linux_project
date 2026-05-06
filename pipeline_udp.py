"""
UDP pipeline：FPGA UDP → ROI裁剪 → LPRNet自动识别（蓝牌/绿牌）
FPGA已做定位，ROI坐标由第241号包携带，两个LPRNet模型同时跑，自动选最优结果。
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
BASE       = '/home/cat/cat_linux_project'
LPR_BLUE   = f'{BASE}/RKNN_NEW/blue_re_run1.rknn'
LPR_GREEN  = f'{BASE}/RKNN_NEW/green_re_run1.rknn'
FONT_PATH  = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
# ══════════════════════════════════════════════════════════════════════════════

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
# 蓝牌7位，绿牌8位（新能源）
_EXPECTED_LEN = {'BLUE': 7, 'GREEN': 8}


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


def _trim_plate(text, color_tag):
    """裁剪到期望长度，优先保留省份+字母开头的前N位。"""
    expected = _EXPECTED_LEN.get(color_tag, 7)
    if len(text) <= expected:
        return text
    # 超长时从头截取期望长度
    return text[:expected]


def plate_format_score(text):
    s = 0
    if len(text) in (7, 8): s += 1
    if len(text) >= 1 and text[0] in _PROVINCE: s += 1
    if len(text) >= 2 and text[1] in _LETTERS: s += 1
    return s


def detect_plate_color(crop_bgr):
    """Detect plate color from crop using HSV. Returns 'BLUE', 'GREEN', or None."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]

    # 蓝牌：H 100~130, S>80, V>60
    blue_mask  = cv2.inRange(hsv, (100, 80, 60), (130, 255, 255))
    # 绿牌：H 40~85, S>60, V>60
    green_mask = cv2.inRange(hsv, (40, 60, 60), (85, 255, 255))

    blue_ratio  = cv2.countNonZero(blue_mask)  / total
    green_ratio = cv2.countNonZero(green_mask) / total

    if blue_ratio > 0.10 and blue_ratio > green_ratio * 1.5:
        return 'BLUE'
    if green_ratio > 0.10 and green_ratio > blue_ratio * 1.5:
        return 'GREEN'
    return None  # 颜色不确定，fallback 到置信度


def lpr_infer_best(crop_bgr, lpr_blue, lpr_green):
    inp   = cv2.resize(crop_bgr, (94, 24))[np.newaxis, :]
    out_b = lpr_blue.inference(inputs=[inp])
    out_g = lpr_green.inference(inputs=[inp])
    tb, cb = lpr_decode(out_b[0])
    tg, cg = lpr_decode(out_g[0])

    color_hint = detect_plate_color(crop_bgr)
    if color_hint == 'BLUE':
        return _trim_plate(tb, 'BLUE'), cb, 'BLUE'
    if color_hint == 'GREEN':
        return _trim_plate(tg, 'GREEN'), cg, 'GREEN'

    sb = cb + plate_format_score(tb) * 0.5
    sg = cg + plate_format_score(tg) * 0.5
    if sg >= sb:
        return _trim_plate(tg, 'GREEN'), cg, 'GREEN'
    return _trim_plate(tb, 'BLUE'), cb, 'BLUE'


_draw_font = None

def _get_draw_font():
    global _draw_font
    if _draw_font is None:
        try:    _draw_font = ImageFont.truetype(FONT_PATH, size=18)
        except: _draw_font = ImageFont.load_default()
    return _draw_font


def draw_result(img_bgr, x1, y1, x2, y2, plate_text, color_tag, conf):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    font  = _get_draw_font()
    color = (0, 220, 0) if color_tag == 'BLUE' else (0, 200, 100)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    draw.text((x1, max(y1 - 22, 0)),
              f'[{color_tag}] {plate_text}  {conf:.2f}',
              font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    print('--> Loading LPRNet models')
    lpr_blue  = RKNNLite(verbose=False)
    lpr_blue.load_rknn(LPR_BLUE);  lpr_blue.init_runtime()
    lpr_green = RKNNLite(verbose=False)
    lpr_green.load_rknn(LPR_GREEN); lpr_green.init_runtime()
    print('done')

    win = 'License Plate Recognition (FPGA UDP)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    inf_fps = 0.0
    t_last  = time.time()

    # 时序平滑：保留最近 N 帧结果，取出现最多的那个
    SMOOTH_N   = 3
    history    = []   # list of (plate, color_tag)
    last_plate = ''
    last_tag   = 'BLUE'
    last_conf  = 0.0

    print('Press q or ESC to quit')
    import queue as _queue
    frame_q = _queue.Queue(maxsize=1)

    def recv_loop(cam):
        while True:
            result = cam.read_latest(timeout=0.5)
            if result is None:
                continue
            if frame_q.full():
                try: frame_q.get_nowait()
                except _queue.Empty: pass
            frame_q.put(result)

    with UDPCamera() as cam:
        import threading as _threading
        _threading.Thread(target=recv_loop, args=(cam,), daemon=True).start()
        while True:
            try:
                frame, (x1, y1, x2, y2), cap_fps = frame_q.get(timeout=0.5)
            except _queue.Empty:
                continue

            vis = frame.copy()

            xmin = max(min(x1, x2), 0); xmax = min(max(x1, x2), WIDTH)
            ymin = max(min(y1, y2), 0); ymax = min(max(y1, y2), HEIGHT)

            if xmax > xmin and ymax > ymin:
                crop = frame[ymin:ymax, xmin:xmax]
                if crop.size > 0:
                    plate, lpr_conf, color_tag = lpr_infer_best(crop, lpr_blue, lpr_green)

                    history.append((plate, color_tag, lpr_conf))
                    if len(history) > SMOOTH_N:
                        history.pop(0)

                    # 选出现次数最多的 (plate, color_tag) 组合
                    from collections import Counter
                    best, _ = Counter((p, t) for p, t, _ in history).most_common(1)[0]
                    best_plate, best_tag = best
                    best_conf = max(c for p, t, c in history if (p, t) == best)

                    if (best_plate, best_tag) != (last_plate, last_tag):
                        print(f'[{best_tag}] {best_plate}  lpr={best_conf:.3f}')
                        last_plate, last_tag, last_conf = best_plate, best_tag, best_conf

                    vis = draw_result(vis, xmin, ymin, xmax, ymax,
                                      last_plate, last_tag, last_conf)

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
    lpr_blue.release()
    lpr_green.release()


if __name__ == '__main__':
    main()

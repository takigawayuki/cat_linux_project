"""
中文车牌识别系统 v2
左侧：车牌类型选择（蓝牌/绿牌）+ 输入源 + 控制
右侧：视频画面（带检测框）
蓝牌：FPGA ROI → LPRNet
绿牌：FPGA UDP → YOLO定位 → LPRNet
"""
import sys
import os
import time
import threading
import queue
from collections import Counter

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont as PILFont

sys.path.append('/home/cat/cat_linux_project')
sys.path.append('/home/cat/cat_linux_project/rknn_model_zoo')

from Drivers.udp_camera import UDPCamera, WIDTH, HEIGHT

# ── 常量 ──────────────────────────────────────────────────────────────────────
BASE       = '/home/cat/cat_linux_project'
LPR_BLUE   = f'{BASE}/RKNN_NEW/blue_re_run1.rknn'
LPR_GREEN  = f'{BASE}/RKNN_NEW/green_re_run1.rknn'
YOLO_MODEL = f'{BASE}/RKNN_NEW/yolov8n_notnms.rknn'
FONT_PATH  = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
TARGET     = 'rk3568'
OBJ_THRESH = 0.70
NMS_THRESH = 0.35
IMG_SIZE   = (640, 640)

SMOOTH_N       = 5
ROI_ALPHA      = 0.3
DISP_W, DISP_H = 640, 480

BG          = '#1e2330'
CARD        = '#252b3b'
ACC         = '#4f8ef7'
GREEN_COLOR = '#4caf50'
FG          = '#e0e6f0'
DIM         = '#7a8499'

CHARS = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
         '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
         '桂','琼','川','贵','云','藏','陕','甘','青','宁','新',
         '0','1','2','3','4','5','6','7','8','9',
         'A','B','C','D','E','F','G','H','J','K',
         'L','M','N','P','Q','R','S','T','U','V',
         'W','X','Y','Z','-']

# ── 字体缓存 ──────────────────────────────────────────────────────────────────
_font_cache: dict = {}

def _get_font(size: int):
    if size not in _font_cache:
        _font_cache[size] = PILFont.truetype(FONT_PATH, size=size)
    return _font_cache[size]

_draw_font = None
def _get_draw_font():
    global _draw_font
    if _draw_font is None:
        try:    _draw_font = PILFont.truetype(FONT_PATH, size=18)
        except: _draw_font = PILFont.load_default()
    return _draw_font


# ── PIL 文字工具 ──────────────────────────────────────────────────────────────
def _pil_label(parent, text, size=11, color=FG, bg=BG, pad_x=4, pad_y=2):
    font  = _get_font(size)
    bbox  = font.getbbox(text or ' ')
    w = max(bbox[2] - bbox[0] + pad_x * 2, 4)
    h = max(bbox[3] - bbox[1] + pad_y * 2, 4)
    img   = Image.new('RGB', (w, h), bg)
    ImageDraw.Draw(img).text((pad_x, pad_y - bbox[1]), text, font=font, fill=color)
    imgtk = ImageTk.PhotoImage(img)
    lbl   = tk.Label(parent, image=imgtk, bg=bg, bd=0, highlightthickness=0)
    lbl._imgtk = imgtk
    return lbl

def _pil_imgtk(text, size=11, color=FG, bg=BG, pad_x=4, pad_y=2):
    font  = _get_font(size)
    bbox  = font.getbbox(text or ' ')
    w = max(bbox[2] - bbox[0] + pad_x * 2, 4)
    h = max(bbox[3] - bbox[1] + pad_y * 2, 4)
    img   = Image.new('RGB', (w, h), bg)
    ImageDraw.Draw(img).text((pad_x, pad_y - bbox[1]), text, font=font, fill=color)
    return ImageTk.PhotoImage(img)


# ── 推理工具函数 ──────────────────────────────────────────────────────────────
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
    expected = 7 if color_tag == 'BLUE' else 8
    return text[:expected] if len(text) > expected else text

def _vote_best(history, expected_len):
    """从历史记录中投票选最优结果。
    优先选长度等于 expected_len 的结果；若没有则选最长的。
    """
    correct = [(p, c) for p, c in history if len(p) == expected_len]
    pool    = correct if correct else history
    if not pool:
        return '', 0.0
    best, _ = Counter(p for p, _ in pool).most_common(1)[0]
    best_conf = max(c for p, c in pool if p == best)
    return best, best_conf

def nms_boxes(boxes, scores):
    x = boxes[:, 0]; y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]; h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep  = []
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
    out = np.squeeze(outputs[0])
    cx, cy, w, h, conf = out[0], out[1], out[2], out[3], out[4]
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    boxes  = np.stack([x1, y1, x2, y2], axis=1)
    mask   = conf > OBJ_THRESH
    boxes, scores = boxes[mask], conf[mask]
    if len(boxes) == 0:
        return None, None
    keep = nms_boxes(boxes, scores)
    return boxes[keep], scores[keep]

def draw_box_blue(img_bgr, x1, y1, x2, y2, plate, conf):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil     = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil)
    font    = _get_draw_font()
    draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=2)
    draw.text((x1, max(y1-22, 0)), f'[BLUE] {plate}  {conf:.2f}',
              font=font, fill=(0, 220, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_box_green(img_bgr, x1, y1, x2, y2, plate, det_score, lpr_conf):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil     = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil)
    font    = _get_draw_font()
    draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 100), width=2)
    draw.text((x1, max(y1-22, 0)),
              f'[GREEN] {plate}  det={det_score:.2f} lpr={lpr_conf:.2f}',
              font=font, fill=(0, 200, 100))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── 主界面 ────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('中文车牌识别系统')
        self.resizable(False, False)
        self.configure(bg=BG)

        self._plate_mode   = tk.StringVar(value='blue')
        self._source       = tk.StringVar(value='camera')
        self._running      = False
        self._models_ready = False
        self._lpr_blue     = None
        self._lpr_green    = None
        self._yolo         = None
        self._cam          = None
        self._coco_helper  = None
        self._history      = []
        self._last_plate   = ''
        self._last_conf    = 0.0
        self._last_roi     = (0, 0, 0, 0)
        self._img_full_path = ''
        self._ui_plate_key  = None
        self._canvas_pending = False

        self._build_ui()
        self._load_models_async()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── UI 构建 ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        left = tk.Frame(self, bg=BG, width=230)
        left.pack(side='left', fill='y', padx=(16, 8), pady=16)
        left.pack_propagate(False)

        _pil_label(left, '中文车牌识别系统', size=11, color=ACC, bg=BG).pack(pady=(0, 14))

        # ── 车牌类型选择 ──────────────────────────────────────────────────────
        mode_card = tk.Frame(left, bg=CARD)
        mode_card.pack(fill='x', pady=(0, 10))
        _pil_label(mode_card, '车牌类型', size=10, color=DIM, bg=CARD).pack(
            anchor='w', padx=10, pady=(8, 4))

        for val, label, color in [('blue', '蓝牌', ACC),
                                   ('green', '绿牌', GREEN_COLOR)]:
            row = tk.Frame(mode_card, bg=CARD)
            row.pack(anchor='w', padx=10, pady=3)
            rb = tk.Radiobutton(row, variable=self._plate_mode, value=val,
                                bg=CARD, fg=color, selectcolor=CARD,
                                activebackground=CARD, bd=0,
                                highlightthickness=0,
                                command=self._on_mode_change)
            rb.pack(side='left')
            _pil_label(row, label, size=10, color=color, bg=CARD).pack(side='left')

        # 当前模式指示
        self._mode_imgtk = _pil_imgtk('模式：蓝牌', size=10, color=ACC, bg=CARD)
        self._lbl_mode = tk.Label(mode_card, image=self._mode_imgtk,
                                  bg=CARD, bd=0, highlightthickness=0)
        self._lbl_mode.pack(anchor='w', padx=10, pady=(2, 8))

        # ── 输入源 ────────────────────────────────────────────────────────────
        src_card = tk.Frame(left, bg=CARD)
        src_card.pack(fill='x', pady=(0, 10))
        _pil_label(src_card, '输入源', size=10, color=DIM, bg=CARD).pack(
            anchor='w', padx=10, pady=(8, 4))

        for val, label in [('camera', '实时摄像头'), ('image', '选择图片')]:
            row = tk.Frame(src_card, bg=CARD)
            row.pack(anchor='w', padx=10, pady=2)
            rb = tk.Radiobutton(row, variable=self._source, value=val,
                                bg=CARD, fg=FG, selectcolor=CARD,
                                activebackground=CARD, bd=0,
                                highlightthickness=0,
                                command=self._on_source_change)
            rb.pack(side='left')
            _pil_label(row, label, size=10, color=FG, bg=CARD).pack(side='left')

        browse_frame = tk.Frame(src_card, bg='#2e3650', cursor='hand2')
        browse_frame.pack(fill='x', padx=10, pady=(4, 6))
        self._browse_lbl = _pil_label(browse_frame, '浏览...', size=10,
                                       color=DIM, bg='#2e3650', pad_x=8, pad_y=4)
        self._browse_lbl.pack()
        self._browse_frame = browse_frame
        browse_frame.bind('<Button-1>', lambda e: self._browse_image())
        self._browse_lbl.bind('<Button-1>', lambda e: self._browse_image())
        self._browse_enabled = False

        self._imgpath_var = tk.StringVar(value='')
        tk.Label(src_card, textvariable=self._imgpath_var,
                 bg=CARD, fg=DIM, font=('fixed', 8),
                 wraplength=200, justify='left').pack(
            anchor='w', padx=10, pady=(0, 8))

        # ── 模型状态 ──────────────────────────────────────────────────────────
        self._model_imgtk = _pil_imgtk('模型加载中...', size=10, color=DIM, bg=BG)
        self._lbl_model = tk.Label(left, image=self._model_imgtk,
                                   bg=BG, bd=0, highlightthickness=0)
        self._lbl_model.pack(pady=(0, 8))

        # ── 开始/停止按钮 ─────────────────────────────────────────────────────
        btn_frame = tk.Frame(left, bg='#3a4060', cursor='hand2')
        btn_frame.pack(fill='x', pady=(0, 14))
        self._run_imgtk = _pil_imgtk('开始运行', size=13, color=DIM,
                                      bg='#3a4060', pad_x=10, pad_y=10)
        self._lbl_run = tk.Label(btn_frame, image=self._run_imgtk,
                                 bg='#3a4060', bd=0, highlightthickness=0)
        self._lbl_run.pack(fill='x')
        self._btn_frame = btn_frame
        btn_frame.bind('<Button-1>', lambda e: self._toggle_run())
        self._lbl_run.bind('<Button-1>', lambda e: self._toggle_run())
        self._run_enabled = False

        # ── 识别结果卡片 ──────────────────────────────────────────────────────
        res_card = tk.Frame(left, bg=CARD)
        res_card.pack(fill='x')
        _pil_label(res_card, '识别结果', size=10, color=DIM, bg=CARD).pack(
            anchor='w', padx=10, pady=(8, 2))

        self._plate_imgtk = _pil_imgtk('—', size=26, color=FG, bg=CARD,
                                        pad_x=6, pad_y=4)
        self._lbl_plate = tk.Label(res_card, image=self._plate_imgtk,
                                   bg=CARD, bd=0, highlightthickness=0)
        self._lbl_plate.pack(pady=(4, 2))

        self._type_imgtk = _pil_imgtk(' ', size=10, color=DIM, bg=CARD)
        self._lbl_type = tk.Label(res_card, image=self._type_imgtk,
                                  bg=CARD, bd=0, highlightthickness=0)
        self._lbl_type.pack()

        tk.Frame(res_card, bg='#3a4060', height=1).pack(fill='x', padx=10, pady=6)

        score_row = tk.Frame(res_card, bg=CARD)
        score_row.pack(fill='x', padx=10, pady=(0, 4))
        _pil_label(score_row, 'Score', size=10, color=DIM, bg=CARD).pack(side='left')
        self._lbl_score = tk.Label(score_row, text='—', bg=CARD, fg=ACC,
                                   font=('fixed', 9))
        self._lbl_score.pack(side='right')

        tk.Frame(res_card, bg='#3a4060', height=1).pack(fill='x', padx=10, pady=4)

        _pil_label(res_card, '车牌位置', size=10, color=DIM, bg=CARD).pack(
            anchor='w', padx=10)
        pos_grid = tk.Frame(res_card, bg=CARD)
        pos_grid.pack(fill='x', padx=10, pady=(4, 10))
        for i, lbl in enumerate(['x1', 'y1', 'x2', 'y2']):
            tk.Label(pos_grid, text=lbl+':', bg=CARD, fg=DIM,
                     font=('fixed', 9), width=3, anchor='e').grid(
                row=i//2, column=(i%2)*2, sticky='e', padx=(0,2), pady=1)
        self._lbl_x1 = tk.Label(pos_grid, text='—', bg=CARD, fg=FG,
                                 font=('fixed', 9), width=5, anchor='w')
        self._lbl_y1 = tk.Label(pos_grid, text='—', bg=CARD, fg=FG,
                                 font=('fixed', 9), width=5, anchor='w')
        self._lbl_x2 = tk.Label(pos_grid, text='—', bg=CARD, fg=FG,
                                 font=('fixed', 9), width=5, anchor='w')
        self._lbl_y2 = tk.Label(pos_grid, text='—', bg=CARD, fg=FG,
                                 font=('fixed', 9), width=5, anchor='w')
        self._lbl_x1.grid(row=0, column=1, sticky='w')
        self._lbl_y1.grid(row=0, column=3, sticky='w')
        self._lbl_x2.grid(row=1, column=1, sticky='w')
        self._lbl_y2.grid(row=1, column=3, sticky='w')

        self._lbl_fps = tk.Label(left, text='', bg=BG, fg=DIM, font=('fixed', 8))
        self._lbl_fps.pack(pady=(10, 0))

        # ── 右侧画面 ──────────────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side='left', fill='both', expand=True, padx=(0, 16), pady=16)
        self._canvas = tk.Label(right, bg='#0d1117', width=DISP_W, height=DISP_H,
                                relief='flat', bd=0, highlightthickness=0)
        self._canvas.pack()
        self._show_placeholder()

    # ── 占位图 ────────────────────────────────────────────────────────────────
    def _show_placeholder(self):
        img  = Image.new('RGB', (DISP_W, DISP_H), '#0d1117')
        draw = ImageDraw.Draw(img)
        draw.text((DISP_W//2 - 80, DISP_H//2 - 12), '等待输入源...',
                  font=_get_font(20), fill='#3a4060')
        self._imgtk = ImageTk.PhotoImage(img)
        self._canvas.configure(image=self._imgtk)

    # ── 模型加载 ──────────────────────────────────────────────────────────────
    def _load_models_async(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            from rknnlite.api import RKNNLite
            from py_utils.rknn_executor import RKNN_model_container
            from py_utils.coco_utils import COCO_test_helper

            self.after(0, lambda: self._set_model_label('加载蓝牌模型...', DIM))
            lb = RKNNLite(verbose=False)
            lb.load_rknn(LPR_BLUE); lb.init_runtime()
            self._lpr_blue = lb

            self.after(0, lambda: self._set_model_label('加载绿牌模型...', DIM))
            lg = RKNNLite(verbose=False)
            lg.load_rknn(LPR_GREEN); lg.init_runtime()
            self._lpr_green = lg

            self.after(0, lambda: self._set_model_label('加载 YOLO 模型...', DIM))
            self._yolo = RKNN_model_container(YOLO_MODEL, TARGET, None)
            self._coco_helper = COCO_test_helper(enable_letter_box=True)

            self._models_ready = True
            self.after(0, self._on_models_ready)
        except Exception as e:
            self.after(0, lambda: self._set_model_label(f'加载失败: {e}', '#f44336'))

    def _on_models_ready(self):
        self._set_model_label('模型已就绪 [OK]', GREEN_COLOR)
        self._set_run_state(True, running=False)

    def _set_model_label(self, text, color):
        self._model_imgtk = _pil_imgtk(text, size=10, color=color, bg=BG)
        self._lbl_model.configure(image=self._model_imgtk)

    # ── 按钮状态 ──────────────────────────────────────────────────────────────
    def _set_run_state(self, enabled, running=False):
        if not enabled:
            bg, text, color = '#3a4060', '开始运行', DIM
        elif running:
            bg, text, color = '#e05555', '停止运行', 'white'
        else:
            bg, text, color = ACC, '开始运行', 'white'
        self._btn_frame.configure(bg=bg)
        self._run_imgtk = _pil_imgtk(text, size=13, color=color,
                                      bg=bg, pad_x=10, pad_y=10)
        self._lbl_run.configure(image=self._run_imgtk, bg=bg)
        self._run_enabled = enabled

    # ── 控件回调 ──────────────────────────────────────────────────────────────
    def _on_mode_change(self):
        if self._running:
            return  # 运行中不允许切换
        mode = self._plate_mode.get()
        if mode == 'blue':
            self._mode_imgtk = _pil_imgtk('模式：蓝牌', size=10, color=ACC, bg=CARD)
        else:
            self._mode_imgtk = _pil_imgtk('模式：绿牌', size=10, color=GREEN_COLOR, bg=CARD)
        self._lbl_mode.configure(image=self._mode_imgtk)

    def _on_source_change(self):
        enabled = self._source.get() == 'image'
        color   = FG if enabled else DIM
        bg      = '#2e3650' if enabled else '#252b3b'
        self._browse_frame.configure(bg=bg)
        self._browse_lbl.configure(bg=bg)
        self._browse_lbl._imgtk = _pil_imgtk('浏览...', size=10, color=color,
                                               bg=bg, pad_x=8, pad_y=4)
        self._browse_lbl.configure(image=self._browse_lbl._imgtk)
        self._browse_enabled = enabled

    def _browse_image(self):
        if not self._browse_enabled:
            return
        path = filedialog.askopenfilename(
            filetypes=[('图片文件', '*.jpg *.jpeg *.png *.bmp')])
        if path:
            self._imgpath_var.set(os.path.basename(path))
            self._img_full_path = path

    def _toggle_run(self):
        if not self._run_enabled:
            return
        if self._running:
            self._running = False
            self._set_run_state(True, running=False)
        else:
            if not self._models_ready:
                return
            self._running = True
            self._history.clear()
            self._ui_plate_key = None
            self._set_run_state(True, running=True)
            if self._source.get() == 'camera':
                threading.Thread(target=self._run_camera, daemon=True).start()
            else:
                threading.Thread(target=self._run_image, daemon=True).start()

    # ── 图片模式 ──────────────────────────────────────────────────────────────
    def _run_image(self):
        path = self._img_full_path
        if not path or not os.path.exists(path):
            self._running = False
            self.after(0, lambda: self._set_run_state(True, running=False))
            return
        img = cv2.imread(path)
        if img is None:
            self._running = False
            self.after(0, lambda: self._set_run_state(True, running=False))
            return

        h, w = img.shape[:2]
        mode = self._plate_mode.get()

        if mode == 'blue':
            plate, conf = lpr_decode(
                self._lpr_blue.inference(inputs=[cv2.resize(img, (94, 24))[np.newaxis, :]])[0])
            plate = _trim_plate(plate, 'BLUE')
            vis   = draw_box_blue(img, 0, 0, w-1, h-1, plate, conf)
            self.after(0, lambda: self._update_result(plate, 'BLUE', conf, 0, 0, w-1, h-1))
        else:
            img_lb   = self._coco_helper.letter_box(
                im=img.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
            img_lb   = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            yolo_out = self._yolo.run([img_lb[np.newaxis, :]])
            boxes, scores = yolo_post_process(yolo_out)
            vis = img.copy()
            if boxes is not None:
                real_boxes = self._coco_helper.get_real_box(boxes)
                for box, score in zip(real_boxes, scores):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w), min(y2, h)
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    out   = self._lpr_green.inference(
                        inputs=[cv2.resize(crop, (94, 24))[np.newaxis, :]])
                    plate, lpr_conf = lpr_decode(out[0])
                    plate = _trim_plate(plate, 'GREEN')
                    vis   = draw_box_green(vis, x1, y1, x2, y2, plate, score, lpr_conf)
                    self.after(0, lambda p=plate, c=lpr_conf, bx=(x1,y1,x2,y2):
                               self._update_result(p, 'GREEN', c, *bx))

        self._show_frame(vis)
        self._running = False
        self.after(0, lambda: self._set_run_state(True, running=False))

    # ── 摄像头模式 ────────────────────────────────────────────────────────────
    def _run_camera(self):
        frame_q = queue.Queue(maxsize=1)

        def recv_loop(cam):
            while self._running:
                result = cam.read_latest(timeout=0.5)
                if result is None:
                    continue
                if frame_q.full():
                    try: frame_q.get_nowait()
                    except queue.Empty: pass
                frame_q.put(result)

        inf_fps    = 0.0
        t_last     = time.time()
        roi_smooth = None

        if self._cam is None:
            self._cam = UDPCamera()
        try:
            cam = self._cam
            threading.Thread(target=recv_loop, args=(cam,), daemon=True).start()
            mode = self._plate_mode.get()
            while self._running:
                try:
                    frame, (x1, y1, x2, y2), cap_fps = frame_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if mode == 'blue':
                    vis, plate, conf, roi, roi_smooth = self._infer_blue(
                        frame, x1, y1, x2, y2, roi_smooth)
                else:
                    vis, plate, conf, roi = self._infer_green(frame)

                t_now   = time.time()
                inf_fps = inf_fps * 0.9 + (1.0 / max(t_now - t_last, 1e-6)) * 0.1
                t_last  = t_now

                self.after(0, lambda p=plate, cf=conf, r=roi,
                           fps=inf_fps, cfps=cap_fps:
                           self._refresh_ui(p, mode, cf, r, fps, cfps))
                self._show_frame(vis)

        except Exception as e:
            print(f'Camera error: {e}')
        finally:
            self._running = False
            self.after(0, lambda: self._set_run_state(True, running=False))

    def _infer_blue(self, frame, x1, y1, x2, y2, roi_smooth):
        raw = (float(x1), float(y1), float(x2), float(y2))
        if roi_smooth is None:
            roi_smooth = raw
        else:
            roi_smooth = tuple(ROI_ALPHA * r + (1 - ROI_ALPHA) * s
                               for r, s in zip(raw, roi_smooth))
        sx1, sy1, sx2, sy2 = (int(v) for v in roi_smooth)
        xmin = max(min(sx1, sx2), 0); xmax = min(max(sx1, sx2), WIDTH)
        ymin = max(min(sy1, sy2), 0); ymax = min(max(sy1, sy2), HEIGHT)

        vis   = frame.copy()
        plate = ''; conf = 0.0
        if xmax > xmin and ymax > ymin:
            crop = frame[ymin:ymax, xmin:xmax]
            if crop.size > 0:
                out   = self._lpr_blue.inference(
                    inputs=[cv2.resize(crop, (94, 24))[np.newaxis, :]])
                plate, conf = lpr_decode(out[0])
                plate = _trim_plate(plate, 'BLUE')
                self._history.append((plate, conf))
                if len(self._history) > SMOOTH_N: self._history.pop(0)
                plate, conf = _vote_best(self._history, expected_len=7)
                vis = draw_box_blue(vis, xmin, ymin, xmax, ymax, plate, conf)
        return vis, plate, conf, (xmin, ymin, xmax, ymax), roi_smooth

    def _infer_green(self, frame):
        img_lb   = self._coco_helper.letter_box(
            im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img_lb   = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        yolo_out = self._yolo.run([img_lb[np.newaxis, :]])
        boxes, scores = yolo_post_process(yolo_out)

        vis   = frame.copy()
        plate = ''; conf = 0.0; roi = (0, 0, 0, 0)
        if boxes is not None:
            real_boxes = self._coco_helper.get_real_box(boxes)
            for box, score in zip(real_boxes, scores):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                out = self._lpr_green.inference(
                    inputs=[cv2.resize(crop, (94, 24))[np.newaxis, :]])
                p, c = lpr_decode(out[0])
                p = _trim_plate(p, 'GREEN')
                self._history.append((p, c))
                if len(self._history) > SMOOTH_N: self._history.pop(0)
                best, best_conf = _vote_best(self._history, expected_len=8)
                plate, conf, roi = best, best_conf, (x1, y1, x2, y2)
                vis = draw_box_green(vis, x1, y1, x2, y2, plate, score, conf)
        return vis, plate, conf, roi

    # ── UI 更新 ───────────────────────────────────────────────────────────────
    def _refresh_ui(self, plate, mode, conf, roi, inf_fps, cap_fps):
        tag = 'BLUE' if mode == 'blue' else 'GREEN'
        self._update_result(plate, tag, conf, *roi)
        self._lbl_fps.configure(
            text=f'CAP {cap_fps:.1f} fps  |  INF {inf_fps:.1f} fps')

    def _update_result(self, plate, tag, conf, x1, y1, x2, y2):
        color_hex = ACC if tag == 'BLUE' else GREEN_COLOR
        key = (plate or '—', tag)
        if key != self._ui_plate_key:
            self._ui_plate_key = key
            self._plate_imgtk = _pil_imgtk(plate or '—', size=26, color=color_hex,
                                            bg=CARD, pad_x=6, pad_y=4)
            self._lbl_plate.configure(image=self._plate_imgtk)
            type_text = '蓝色车牌' if tag == 'BLUE' else '绿色车牌'
            self._type_imgtk = _pil_imgtk(type_text, size=10, color=color_hex, bg=CARD)
            self._lbl_type.configure(image=self._type_imgtk)
        self._lbl_score.configure(text=f'{conf:.3f}' if conf else '—')
        self._lbl_x1.configure(text=str(x1))
        self._lbl_y1.configure(text=str(y1))
        self._lbl_x2.configure(text=str(x2))
        self._lbl_y2.configure(text=str(y2))

    def _show_frame(self, bgr):
        if self._canvas_pending:
            return
        self._canvas_pending = True
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((DISP_W, DISP_H), Image.BILINEAR)
        self.after(0, lambda i=img: self._set_canvas(i))

    def _set_canvas(self, img):
        self._imgtk = ImageTk.PhotoImage(img)
        self._canvas.configure(image=self._imgtk)
        self._canvas_pending = False

    def _on_close(self):
        self._running = False
        if self._cam:       self._cam.release()
        if self._lpr_blue:  self._lpr_blue.release()
        if self._lpr_green: self._lpr_green.release()
        if self._yolo:      self._yolo.release()
        self.destroy()


if __name__ == '__main__':
    app = App()
    app.mainloop()

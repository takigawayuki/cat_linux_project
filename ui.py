"""
中文车牌识别与管理系统
左侧：输入源选择 + 控制按钮
右侧：视频/图片画面（带检测框）
底部：识别结果、Score、车牌位置
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

from pipeline_udp import (lpr_infer_best, draw_result as draw_box)
from Drivers.udp_camera import UDPCamera, WIDTH, HEIGHT

# ── 常量 ──────────────────────────────────────────────────────────────────────
BASE      = '/home/cat/cat_linux_project'
LPR_BLUE  = f'{BASE}/RKNN_NEW/blue_re_run1.rknn'
LPR_GREEN = f'{BASE}/RKNN_NEW/green_re_run1.rknn'
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

SMOOTH_N       = 5          # 投票フレーム数（増やすほど安定、反応は遅くなる）
ROI_ALPHA      = 0.3        # ROI EMA係数（小さいほど平滑、大きいほど追従が速い）
DISP_W, DISP_H = 640, 480

BG   = '#1e2330'
CARD = '#252b3b'
ACC  = '#4f8ef7'
FG   = '#e0e6f0'
DIM  = '#7a8499'
GREEN_COLOR = '#4caf50'

# フォントキャッシュ（毎フレーム truetype() を呼ばないように）
_font_cache: dict = {}

def _get_font(size: int):
    if size not in _font_cache:
        _font_cache[size] = PILFont.truetype(FONT_PATH, size=size)
    return _font_cache[size]


# ── PIL 文字工具 ──────────────────────────────────────────────────────────────
def _pil_label(parent, text, size=11, color=FG, bg=BG, pad_x=4, pad_y=2):
    """返回一个用 PIL NotoSansCJK 渲染文字的 tk.Label（image模式）。"""
    font = _get_font(size)
    bbox = font.getbbox(text or ' ')
    w = max(bbox[2] - bbox[0] + pad_x * 2, 4)
    h = max(bbox[3] - bbox[1] + pad_y * 2, 4)
    img  = Image.new('RGB', (w, h), bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x, pad_y - bbox[1]), text, font=font, fill=color)
    imgtk = ImageTk.PhotoImage(img)
    lbl = tk.Label(parent, image=imgtk, bg=bg, bd=0, highlightthickness=0)
    lbl._imgtk = imgtk  # 防止 GC
    return lbl


def _pil_imgtk(text, size=11, color=FG, bg=BG, pad_x=4, pad_y=2):
    """返回 PIL 渲染的 ImageTk（用于需要动态更新的 Label）。"""
    font = _get_font(size)
    bbox = font.getbbox(text or ' ')
    w = max(bbox[2] - bbox[0] + pad_x * 2, 4)
    h = max(bbox[3] - bbox[1] + pad_y * 2, 4)
    img  = Image.new('RGB', (w, h), bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x, pad_y - bbox[1]), text, font=font, fill=color)
    return ImageTk.PhotoImage(img)


# ── 主界面 ────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('中文车牌识别与管理系统')
        self.resizable(False, False)
        self.configure(bg=BG)

        self._models_loaded = False
        self._lpr_blue  = None
        self._lpr_green = None
        self._running   = False
        self._source    = tk.StringVar(value='camera')
        self._history   = []
        self._last_plate = ''
        self._last_tag   = 'BLUE'
        self._last_conf  = 0.0
        self._last_roi   = (0, 0, 0, 0)
        self._img_full_path = ''
        self._ui_plate_key  = None   # (plate, tag) キャッシュ、変化時のみ再描画

        self._build_ui()
        self._load_models_async()
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._canvas_pending = False

    # ── UI 构建 ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── 左侧面板 ──────────────────────────────────────────────────────────
        left = tk.Frame(self, bg=BG, width=220)
        left.pack(side='left', fill='y', padx=(16, 8), pady=16)
        left.pack_propagate(False)

        _pil_label(left, '中文车牌识别与管理系统',
                   size=11, color=ACC, bg=BG).pack(pady=(0, 18))

        # 输入源卡片
        src_card = tk.Frame(left, bg=CARD, bd=0)
        src_card.pack(fill='x', pady=(0, 12))
        _pil_label(src_card, '输入源', size=10, color=DIM, bg=CARD).pack(
            anchor='w', padx=10, pady=(8, 4))

        # 单选按钮行（PIL标签 + tkinter Radiobutton indicator）
        for val, label in [('camera', '实时摄像头'), ('image', '选择图片')]:
            row = tk.Frame(src_card, bg=CARD)
            row.pack(anchor='w', padx=10, pady=2)
            rb = tk.Radiobutton(row, variable=self._source, value=val,
                                bg=CARD, fg=FG, selectcolor=CARD,
                                activebackground=CARD, bd=0,
                                highlightthickness=0,
                                command=self._on_source_change)
            rb.pack(side='left')
            _pil_label(row, label, size=11, color=FG, bg=CARD).pack(side='left')

        # 浏览按钮（PIL渲染文字 + 点击绑定）
        browse_frame = tk.Frame(src_card, bg='#2e3650', cursor='hand2')
        browse_frame.pack(fill='x', padx=10, pady=(4, 6))
        self._browse_lbl = _pil_label(browse_frame, '浏览...', size=10,
                                       color=FG, bg='#2e3650', pad_x=8, pad_y=4)
        self._browse_lbl.pack()
        self._browse_frame = browse_frame
        browse_frame.bind('<Button-1>', lambda e: self._browse_image())
        self._browse_lbl.bind('<Button-1>', lambda e: self._browse_image())
        self._set_browse_state(False)

        self._imgpath_var = tk.StringVar(value='')
        self._lbl_imgpath = tk.Label(src_card, textvariable=self._imgpath_var,
                                     bg=CARD, fg=DIM, font=('fixed', 8),
                                     wraplength=190, justify='left')
        self._lbl_imgpath.pack(anchor='w', padx=10, pady=(0, 8))

        # 模型状态（动态，需要更新）
        self._model_imgtk = _pil_imgtk('模型加载中...', size=10, color=DIM, bg=BG)
        self._lbl_model = tk.Label(left, image=self._model_imgtk, bg=BG,
                                   bd=0, highlightthickness=0)
        self._lbl_model.pack(pady=(0, 10))

        # 开始/停止按钮（PIL渲染文字，背景色可变）
        self._btn_bg = ACC
        btn_frame = tk.Frame(left, bg=ACC, cursor='hand2')
        btn_frame.pack(fill='x', pady=(0, 16))
        self._run_imgtk = _pil_imgtk('开始运行', size=13, color='white',
                                      bg=ACC, pad_x=10, pad_y=10)
        self._lbl_run = tk.Label(btn_frame, image=self._run_imgtk,
                                 bg=ACC, bd=0, highlightthickness=0)
        self._lbl_run.pack(fill='x')
        self._btn_frame = btn_frame
        btn_frame.bind('<Button-1>', lambda e: self._toggle_run())
        self._lbl_run.bind('<Button-1>', lambda e: self._toggle_run())
        self._set_run_state(False)

        # 识别结果卡片
        res_card = tk.Frame(left, bg=CARD, bd=0)
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

        tk.Frame(res_card, bg='#3a4060', height=1).pack(fill='x', padx=10, pady=8)

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
                         row=i//2, column=(i%2)*2, sticky='e', padx=(0, 2), pady=1)
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

        self._lbl_fps = tk.Label(left, text='', bg=BG, fg=DIM,
                                 font=('fixed', 8))
        self._lbl_fps.pack(pady=(12, 0))

        # ── 右侧画面 ──────────────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side='left', fill='both', expand=True, padx=(0, 16), pady=16)

        self._canvas = tk.Label(right, bg='#0d1117', width=DISP_W, height=DISP_H,
                                relief='flat', bd=0, highlightthickness=0)
        self._canvas.pack()
        self._show_placeholder()

    # ── 辅助：按钮状态 ────────────────────────────────────────────────────────
    def _set_browse_state(self, enabled):
        color = FG if enabled else DIM
        bg    = '#2e3650' if enabled else '#252b3b'
        self._browse_frame.configure(bg=bg, cursor='hand2' if enabled else '')
        self._browse_lbl.configure(bg=bg)
        self._browse_lbl._imgtk = _pil_imgtk('浏览...', size=10, color=color,
                                               bg=bg, pad_x=8, pad_y=4)
        self._browse_lbl.configure(image=self._browse_lbl._imgtk)
        self._browse_enabled = enabled

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

    # ── 占位图 ────────────────────────────────────────────────────────────────
    def _show_placeholder(self):
        img  = Image.new('RGB', (DISP_W, DISP_H), '#0d1117')
        draw = ImageDraw.Draw(img)
        font = PILFont.truetype(FONT_PATH, 20)
        draw.text((DISP_W//2 - 80, DISP_H//2 - 12), '等待输入源...', font=font, fill='#3a4060')
        self._imgtk = ImageTk.PhotoImage(img)
        self._canvas.configure(image=self._imgtk)

    # ── 模型加载 ──────────────────────────────────────────────────────────────
    def _load_models_async(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            from rknnlite.api import RKNNLite
            lb = RKNNLite(verbose=False)
            lb.load_rknn(LPR_BLUE);  lb.init_runtime()
            lg = RKNNLite(verbose=False)
            lg.load_rknn(LPR_GREEN); lg.init_runtime()
            self._lpr_blue  = lb
            self._lpr_green = lg
            self._models_loaded = True
            self.after(0, self._on_models_ready)
        except Exception as e:
            self.after(0, lambda: self._update_model_label(
                f'模型加载失败', '#f44336'))

    def _on_models_ready(self):
        self._update_model_label('模型已就绪 [OK]', GREEN_COLOR)
        self._set_run_state(True, running=False)

    def _update_model_label(self, text, color):
        self._model_imgtk = _pil_imgtk(text, size=10, color=color, bg=BG)
        self._lbl_model.configure(image=self._model_imgtk)

    # ── 控件回调 ──────────────────────────────────────────────────────────────
    def _on_source_change(self):
        self._set_browse_state(self._source.get() == 'image')

    def _browse_image(self):
        if not getattr(self, '_browse_enabled', False):
            return
        path = filedialog.askopenfilename(
            filetypes=[('图片文件', '*.jpg *.jpeg *.png *.bmp')])
        if path:
            self._imgpath_var.set(os.path.basename(path))
            self._img_full_path = path

    def _toggle_run(self):
        if not getattr(self, '_run_enabled', False):
            return
        if self._running:
            self._running = False
            self._set_run_state(True, running=False)
        else:
            if not self._models_loaded:
                return
            self._running = True
            self._set_run_state(True, running=True)
            self._history.clear()
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
        plate, conf, tag = lpr_infer_best(img, self._lpr_blue, self._lpr_green)
        vis = draw_box(img, 0, 0, w-1, h-1, plate, tag, conf)
        self.after(0, lambda: self._update_result(plate, tag, conf, 0, 0, w-1, h-1))
        self._show_frame(vis)
        self._running = False
        self.after(0, lambda: self._set_run_state(True, running=False))

    # ── 摄像头模式（UDP）────────────────────────────────────────────────────
    def _run_camera(self):
        # 用 maxsize=1 的队列：推理慢时自动丢弃旧帧，始终处理最新帧
        frame_q = queue.Queue(maxsize=1)

        def recv_loop(cam):
            while self._running:
                result = cam.read_latest(timeout=0.5)
                if result is None:
                    continue
                # 队列满时丢弃旧帧，放入最新帧
                if frame_q.full():
                    try: frame_q.get_nowait()
                    except queue.Empty: pass
                frame_q.put(result)

        inf_fps = 0.0
        t_last  = time.time()
        roi_smooth = None   # EMA平滑后的ROI，(x1,y1,x2,y2) float
        try:
            with UDPCamera() as cam:
                threading.Thread(target=recv_loop, args=(cam,), daemon=True).start()
                while self._running:
                    try:
                        frame, (x1, y1, x2, y2), cap_fps = frame_q.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    # ROI EMA平滑，抑制FPGA定位框跳动
                    raw_roi = (float(x1), float(y1), float(x2), float(y2))
                    if roi_smooth is None:
                        roi_smooth = raw_roi
                    else:
                        roi_smooth = tuple(
                            ROI_ALPHA * r + (1 - ROI_ALPHA) * s
                            for r, s in zip(raw_roi, roi_smooth)
                        )
                    sx1, sy1, sx2, sy2 = (int(v) for v in roi_smooth)

                    vis = frame.copy()
                    xmin = max(min(sx1, sx2), 0); xmax = min(max(sx1, sx2), WIDTH)
                    ymin = max(min(sy1, sy2), 0); ymax = min(max(sy1, sy2), HEIGHT)

                    if xmax > xmin and ymax > ymin:
                        crop = frame[ymin:ymax, xmin:xmax]
                        if crop.size > 0:
                            plate, lpr_conf, color_tag = lpr_infer_best(
                                crop, self._lpr_blue, self._lpr_green)
                            self._history.append((plate, color_tag, lpr_conf))
                            if len(self._history) > SMOOTH_N:
                                self._history.pop(0)
                            best, _ = Counter(
                                (p, t) for p, t, _ in self._history
                            ).most_common(1)[0]
                            best_plate, best_tag = best
                            best_conf = max(
                                c for p, t, c in self._history if (p, t) == best)
                            self._last_plate = best_plate
                            self._last_tag   = best_tag
                            self._last_conf  = best_conf
                            self._last_roi   = (xmin, ymin, xmax, ymax)
                            vis = draw_box(vis, xmin, ymin, xmax, ymax,
                                          best_plate, best_tag, best_conf)

                    t_now   = time.time()
                    inf_fps = inf_fps * 0.9 + (1.0 / max(t_now - t_last, 1e-6)) * 0.1
                    t_last  = t_now

                    self.after(0, lambda p=self._last_plate, tg=self._last_tag,
                               cf=self._last_conf, roi=self._last_roi,
                               fps=inf_fps, cfps=cap_fps:
                               self._refresh_ui(p, tg, cf, roi, fps, cfps))
                    self._show_frame(vis)

        except Exception as e:
            print(f'Camera error: {e}')
        finally:
            self._running = False
            self.after(0, lambda: self._set_run_state(True, running=False))

    # ── UI 更新 ───────────────────────────────────────────────────────────────
    def _refresh_ui(self, plate, tag, conf, roi, inf_fps, cap_fps):
        self._update_result(plate, tag, conf, *roi)
        self._lbl_fps.configure(
            text=f'CAP {cap_fps:.1f} fps  |  INF {inf_fps:.1f} fps')

    def _update_result(self, plate, tag, conf, x1, y1, x2, y2):
        # 车牌文字和颜色变化时才重新渲染 PIL 图像（避免每帧都调 truetype 渲染）
        key = (plate or '—', tag)
        if key != self._ui_plate_key:
            self._ui_plate_key = key
            color_hex = ACC if tag == 'BLUE' else GREEN_COLOR
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
        # 推理线程调用。_canvas_pending 为 True 时说明主线程还没处理完上一帧，直接丢弃。
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
        if self._lpr_blue:  self._lpr_blue.release()
        if self._lpr_green: self._lpr_green.release()
        self.destroy()


if __name__ == '__main__':
    app = App()
    app.mainloop()

#!/usr/bin/env python3
"""RKNN YOLO + LPRNet plate pipeline for a single image or image folder.

The script is intentionally CPU-postprocess heavy: RKNN runs the two neural
networks, while DFL decode, NMS, plate crop, color/type estimation, and
constrained CTC beam search stay in Python for easy validation before C++ port.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
except Exception as exc:  # pragma: no cover - OpenCV is required for real inference.
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None
try:
    import numpy as np
except Exception as exc:  # pragma: no cover - NumPy is required for real inference.
    np = None
    NP_IMPORT_ERROR = exc
else:
    NP_IMPORT_ERROR = None

try:
    from rknnlite.api import RKNNLite
except Exception as exc:  # pragma: no cover - this script normally runs on RKNN boards.
    RKNNLite = None
    RKNN_IMPORT_ERROR = exc
else:
    RKNN_IMPORT_ERROR = None


YOLO_CLASSES = ("plate", "person", "car", "traffic_light")
STRIDES = (8, 16, 32)
IMG_SIZE = (640, 640)  # width, height
LPR_SIZE = (94, 24)    # width, height

# Must match LPRNet_Pytorch/data/load_data.py for unified class_num=74.
CHARS = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "学", "挂", "港", "澳", "使", "领", "警", "临",
    "-",
]
BLANK_INDEX = len(CHARS) - 1
PROVINCES = set(CHARS[:31])
DIGITS = set("0123456789")
LETTERS = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
ALNUM = DIGITS | LETTERS
SPECIALS = set("学挂港澳使领警临")

CCPD_PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学", "O",
]
CCPD_ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "O",
]


@dataclass
class Detection:
    box: np.ndarray  # xyxy in original image coords
    class_id: int
    score: float


@dataclass
class PlateResult:
    box: List[int]
    detection_index: int
    det_score: float
    estimated_type: str
    decoded_type: str
    plate_subtype: str
    plate_type: str
    plate_text: str
    valid: bool
    invalid_reason: str
    lpr_score: float
    beam_score: float
    raw_text: str
    crop_path: Optional[str]
    crop_width: int
    crop_height: int
    gt_text: Optional[str]
    match: Optional[bool]


class DebugWriter:
    def __init__(self, output_dir: Path, args: argparse.Namespace):
        self.debug_dir = output_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.files = []
        self.image_writer = self._csv_writer("images.csv", [
            "image", "gt_text", "width", "height", "detection_count", "plate_count",
            "plate_texts", "best_plate_text", "best_lpr_score", "best_det_score",
            "yolo_pre_ms", "yolo_infer_ms", "yolo_post_ms", "lpr_total_ms",
            "total_ms", "vis_path",
        ])
        self.det_writer = self._csv_writer("detections.csv", [
            "image", "detection_index", "class_id", "class_name", "score",
            "x1", "y1", "x2", "y2", "width", "height", "area",
        ])
        self.plate_writer = self._csv_writer("plates.csv", [
            "image", "plate_index", "detection_index", "gt_text", "match",
            "det_score", "estimated_type", "decoded_type", "plate_subtype", "plate_type",
            "plate_text", "valid", "invalid_reason", "raw_text", "lpr_score", "beam_score", "crop_path",
            "crop_width", "crop_height", "x1", "y1", "x2", "y2",
        ])
        self.candidate_writer = self._csv_writer("decode_candidates.csv", [
            "image", "plate_index", "candidate_rank", "selected", "estimated_type",
            "plate_type", "plate_subtype", "target_len", "text_len", "length_ok",
            "text", "lpr_score", "beam_score",
        ])
        self.results_jsonl = open(self.debug_dir / "results.jsonl", "w", encoding="utf-8")
        self.files.append(self.results_jsonl)
        self.write_config(args)

    def _csv_writer(self, name: str, fieldnames: List[str]) -> csv.DictWriter:
        handle = open(self.debug_dir / name, "w", newline="", encoding="utf-8-sig")
        self.files.append(handle)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        return writer

    def write_config(self, args: argparse.Namespace) -> None:
        config = vars(args).copy()
        config.update({
            "classes": YOLO_CLASSES,
            "chars_len": len(CHARS),
            "blank_index": BLANK_INDEX,
            "img_size": IMG_SIZE,
            "lpr_size": LPR_SIZE,
        })
        with open(self.debug_dir / "run_config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    def write_jsonl(self, payload: Dict) -> None:
        self.results_jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        for handle in self.files:
            handle.flush()

    def close(self) -> None:
        for handle in self.files:
            handle.close()


def image_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.iterdir()):
        if child.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield child


def parse_ccpd_gt(image_path: Path) -> Optional[str]:
    parts = image_path.stem.split("-")
    if len(parts) < 5:
        return None
    try:
        labels = [int(v) for v in parts[4].split("_")]
    except ValueError:
        return None
    if len(labels) < 2:
        return None
    if labels[0] >= len(CCPD_PROVINCES):
        return None
    chars = [CCPD_PROVINCES[labels[0]]]
    for idx in labels[1:]:
        if idx >= len(CCPD_ALPHABETS):
            return None
        chars.append(CCPD_ALPHABETS[idx])
    return "".join(chars)


def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = IMG_SIZE,
              color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    src_h, src_w = im.shape[:2]
    dst_w, dst_h = new_shape
    ratio = min(dst_w / src_w, dst_h / src_h)
    resized_w, resized_h = int(round(src_w * ratio)), int(round(src_h * ratio))
    pad_w = (dst_w - resized_w) / 2
    pad_h = (dst_h - resized_h) / 2

    if (src_w, src_h) != (resized_w, resized_h):
        im = cv2.resize(im, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (pad_w, pad_h)


def preprocess_yolo(frame_bgr: np.ndarray, color_order: str) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    img, ratio, pad = letterbox(frame_bgr, IMG_SIZE)
    if color_order == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[np.newaxis, :].astype(np.uint8), ratio, pad


def to_nchw(arr: np.ndarray, preferred_channels: Sequence[int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[np.newaxis, :]
    if arr.ndim != 4:
        raise ValueError(f"expected 4D output, got shape={arr.shape}")
    if arr.shape[1] in preferred_channels:
        return arr.astype(np.float32, copy=False)
    if arr.shape[-1] in preferred_channels:
        return arr.transpose(0, 3, 1, 2).astype(np.float32, copy=False)
    raise ValueError(f"cannot infer NCHW layout from shape={arr.shape}, channels={preferred_channels}")


def group_yolo_outputs(outputs: Sequence[np.ndarray], num_classes: int) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]]:
    if len(outputs) != 9:
        raise ValueError(f"YOLO RKNN should return 9 outputs, got {len(outputs)}")

    tensors = []
    for idx, out in enumerate(outputs):
        arr = np.asarray(out)
        if arr.ndim == 3:
            arr = arr[np.newaxis, :]
        shape = arr.shape
        channel = shape[1] if shape[1] in (64, num_classes, 1) else shape[-1]
        height = shape[2] if shape[1] in (64, num_classes, 1) else shape[1]
        width = shape[3] if shape[1] in (64, num_classes, 1) else shape[2]
        tensors.append((idx, arr, int(channel), int(height), int(width)))

    groups = []
    for stride in STRIDES:
        expected_h = IMG_SIZE[1] // stride
        expected_w = IMG_SIZE[0] // stride
        same_scale = [t for t in tensors if t[3] == expected_h and t[4] == expected_w]
        bbox = next((t for t in same_scale if t[2] == 64), None)
        cls = next((t for t in same_scale if t[2] == num_classes), None)
        score = next((t for t in same_scale if t[2] == 1), None)
        if bbox is None or cls is None:
            raise ValueError(f"missing bbox/cls output for stride={stride}; outputs={[t[1].shape for t in tensors]}")
        groups.append((
            to_nchw(bbox[1], [64]),
            to_nchw(cls[1], [num_classes]),
            to_nchw(score[1], [1]) if score is not None else None,
            stride,
        ))
    return groups


def dfl_decode(box_logits: np.ndarray) -> np.ndarray:
    # box_logits: [N, 64], 64 = 4 sides * 16 bins.
    bins = box_logits.reshape(-1, 4, 16)
    bins = bins - bins.max(axis=2, keepdims=True)
    probs = np.exp(bins)
    probs /= probs.sum(axis=2, keepdims=True)
    return (probs * np.arange(16, dtype=np.float32)).sum(axis=2)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[np.where(iou <= threshold)[0] + 1]
    return np.asarray(keep, dtype=np.int64)


def postprocess_yolo(outputs: Sequence[np.ndarray], ratio: float, pad: Tuple[float, float],
                     original_shape: Tuple[int, int], conf_thres: float,
                     nms_thres: float) -> List[Detection]:
    boxes_all, scores_all, classes_all = [], [], []
    groups = group_yolo_outputs(outputs, len(YOLO_CLASSES))

    for bbox_feat, cls_feat, cls_sum_feat, stride in groups:
        del cls_sum_feat  # cls_sum is useful for diagnostics; final score uses max(cls).
        _, _, h, w = cls_feat.shape
        cls = cls_feat.reshape(1, len(YOLO_CLASSES), -1)[0].T
        cls_max = cls.max(axis=1)
        mask = cls_max >= conf_thres
        if not np.any(mask):
            continue

        box_logits = bbox_feat.reshape(1, 64, -1)[0].T[mask]
        scores = cls_max[mask]
        classes = np.argmax(cls[mask], axis=1)
        dist = dfl_decode(box_logits)

        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.float32)[mask]
        x1 = (grid[:, 0] + 0.5 - dist[:, 0]) * stride
        y1 = (grid[:, 1] + 0.5 - dist[:, 1]) * stride
        x2 = (grid[:, 0] + 0.5 + dist[:, 2]) * stride
        y2 = (grid[:, 1] + 0.5 + dist[:, 3]) * stride
        boxes_all.append(np.stack([x1, y1, x2, y2], axis=1))
        scores_all.append(scores.astype(np.float32))
        classes_all.append(classes.astype(np.int64))

    if not boxes_all:
        return []

    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    classes = np.concatenate(classes_all, axis=0)

    # Map 640x640 letterbox coords back to original image coords.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
    orig_h, orig_w = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

    detections = []
    for class_id in sorted(set(classes.tolist())):
        inds = np.where(classes == class_id)[0]
        keep = nms_xyxy(boxes[inds], scores[inds], nms_thres)
        for idx in inds[keep]:
            detections.append(Detection(box=boxes[idx], class_id=int(classes[idx]), score=float(scores[idx])))
    detections.sort(key=lambda d: (-d.score, d.class_id))
    return detections


def crop_with_padding(image: np.ndarray, box: Sequence[float], pad_x: float, pad_y: float) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    x1 = int(max(0, math.floor(x1 - bw * pad_x)))
    x2 = int(min(w, math.ceil(x2 + bw * pad_x)))
    y1 = int(max(0, math.floor(y1 - bh * pad_y)))
    y2 = int(min(h, math.ceil(y2 + bh * pad_y)))
    if x2 - x1 < 20 or y2 - y1 < 8:
        return None
    return image[y1:y2, x1:x2].copy()


def estimate_plate_type(crop_bgr: np.ndarray) -> str:
    if crop_bgr.size == 0:
        return "unknown_7"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    area = float(hsv.shape[0] * hsv.shape[1])
    blue = np.count_nonzero((h >= 95) & (h <= 135) & (s >= 50) & (v >= 50)) / area
    green = np.count_nonzero((h >= 35) & (h <= 90) & (s >= 40) & (v >= 45)) / area
    yellow = np.count_nonzero((h >= 15) & (h <= 40) & (s >= 50) & (v >= 70)) / area
    dark = np.count_nonzero(v <= 65) / area
    if green > 0.18 and green >= blue and green >= yellow:
        return "green"
    if yellow > 0.18 and yellow >= blue:
        return "yellow"
    if blue > 0.18:
        return "blue"
    if dark > 0.55:
        return "black"
    return "unknown_7"


def target_len_for_type(plate_type: str, plate_subtype: Optional[str] = None) -> int:
    if plate_type in {"green", "unknown_8"} or plate_subtype == "tractor_green":
        return 8
    return 7


def char_allowed(ch: str, pos: int, plate_type: str, plate_subtype: Optional[str]) -> bool:
    if plate_type == "black":
        return ch != "-"
    if pos == 0:
        return ch in PROVINCES
    if pos == 1:
        if plate_type == "special_7":
            return ch in ALNUM
        return ch in LETTERS
    if plate_subtype == "tractor_green":
        return ch in ALNUM or ch in {"学", "挂"}
    if plate_type == "yellow":
        return ch in ALNUM or ch in {"学", "挂"}
    if plate_type == "special_7":
        return ch in ALNUM or ch in SPECIALS
    return ch in ALNUM


def is_valid_prefix(prefix: Tuple[int, ...], plate_type: str, plate_subtype: Optional[str]) -> bool:
    target_len = target_len_for_type(plate_type, plate_subtype)
    if len(prefix) > target_len:
        return False
    for pos, idx in enumerate(prefix):
        if idx == BLANK_INDEX:
            return False
        if not char_allowed(CHARS[idx], pos, plate_type, plate_subtype):
            return False
    return True


def log_softmax_time(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32, copy=False)
    logits = logits - logits.max(axis=0, keepdims=True)
    logsum = np.log(np.exp(logits).sum(axis=0, keepdims=True) + 1e-12)
    return logits - logsum


def ctc_collapse(indices: Sequence[int]) -> Tuple[int, ...]:
    collapsed: List[int] = []
    prev = BLANK_INDEX
    for idx in indices:
        idx = int(idx)
        if idx == BLANK_INDEX:
            prev = BLANK_INDEX
            continue
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    return tuple(collapsed)


def normalize_lpr_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 3:
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError(f"LPR logits should be 2D/3D, got shape={logits.shape}")
    if logits.shape[0] == len(CHARS):
        return logits
    if logits.shape[1] == len(CHARS):
        return logits.T
    raise ValueError(f"LPR logits class dim should be {len(CHARS)}, got shape={logits.shape}")


def greedy_debug_decode(logits: np.ndarray) -> Tuple[str, float]:
    logits = normalize_lpr_logits(logits)
    log_probs = log_softmax_time(logits)
    raw = np.argmax(logits, axis=0)
    collapsed = ctc_collapse(raw)
    text = "".join(CHARS[i] for i in collapsed)
    conf = float(np.exp(log_probs[raw, np.arange(log_probs.shape[1])]).mean()) if raw.size else 0.0
    return text, conf


def constrained_ctc_decode_one(logits: np.ndarray, plate_type: str, plate_subtype: Optional[str],
                               beam_width: int, beam_topk: int) -> Tuple[str, float, float]:
    logits = normalize_lpr_logits(logits)
    log_probs = log_softmax_time(logits)
    target_len = target_len_for_type(plate_type, plate_subtype)
    beams: Dict[Tuple[int, ...], float] = {tuple(): 0.0}

    for t in range(log_probs.shape[1]):
        top = np.argsort(log_probs[:, t])[-beam_topk:].tolist()
        if BLANK_INDEX not in top:
            top.append(BLANK_INDEX)
        next_beams: Dict[Tuple[int, ...], float] = {}
        for raw_seq, score in beams.items():
            for idx in top:
                new_raw = raw_seq + (int(idx),)
                prefix = ctc_collapse(new_raw)
                if len(prefix) > target_len:
                    continue
                if not is_valid_prefix(prefix, plate_type, plate_subtype):
                    continue
                new_score = score + float(log_probs[idx, t])
                prev_score = next_beams.get(new_raw)
                if prev_score is None or new_score > prev_score:
                    next_beams[new_raw] = new_score
        if not next_beams:
            next_beams = beams
        ordered = sorted(next_beams.items(), key=lambda item: item[1], reverse=True)
        beams = dict(ordered[:beam_width])

    candidates = []
    for raw_seq, score in beams.items():
        prefix = ctc_collapse(raw_seq)
        if len(prefix) == target_len and is_valid_prefix(prefix, plate_type, plate_subtype):
            candidates.append((prefix, score))
    if not candidates:
        for raw_seq, score in beams.items():
            prefix = ctc_collapse(raw_seq)
            if prefix and is_valid_prefix(prefix, plate_type, plate_subtype):
                candidates.append((prefix, score))
    if not candidates:
        return "", 0.0, float("-inf")

    prefix, score = max(candidates, key=lambda item: item[1])
    text = "".join(CHARS[i] for i in prefix)
    norm_log_score = score / max(1, logits.shape[1])
    return text, float(math.exp(norm_log_score)), float(score)


def decode_with_type_fallback(logits: np.ndarray, estimated_type: str, beam_width: int,
                              beam_topk: int) -> Tuple[str, float, float, str, str, List[Dict]]:
    if estimated_type.startswith("unknown"):
        attempts = [
            ("blue", None),
            ("green", None),
            ("yellow", None),
            ("black", None),
            ("special_7", None),
            ("unknown_8", None),
        ]
    else:
        attempts = [(estimated_type, None)]
        if estimated_type == "green":
            attempts.append(("green", "tractor_green"))
        attempts.extend([("blue", None), ("green", None), ("yellow", None), ("black", None)])

    seen = set()
    candidates: List[Dict] = []
    best = ("", 0.0, float("-inf"), estimated_type, "")
    best_idx = -1
    for plate_type, subtype in attempts:
        key = (plate_type, subtype)
        if key in seen:
            continue
        seen.add(key)
        text, prob, score = constrained_ctc_decode_one(logits, plate_type, subtype, beam_width, beam_topk)
        target_len = target_len_for_type(plate_type, subtype)
        candidate = {
            "candidate_rank": len(candidates),
            "selected": False,
            "estimated_type": estimated_type,
            "plate_type": plate_type,
            "plate_subtype": subtype or "",
            "target_len": target_len,
            "text_len": len(text),
            "length_ok": len(text) == target_len,
            "text": text,
            "lpr_score": prob,
            "beam_score": score,
        }
        candidates.append(candidate)
        if score > best[2]:
            best = (text, prob, score, plate_type, subtype or "")
            best_idx = len(candidates) - 1
    if best_idx >= 0:
        candidates[best_idx]["selected"] = True
    return (*best, candidates)


def preprocess_lpr(crop_bgr: np.ndarray, color_order: str) -> np.ndarray:
    img = cv2.resize(crop_bgr, LPR_SIZE, interpolation=cv2.INTER_LINEAR)
    if color_order == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[np.newaxis, :].astype(np.uint8)


def draw_detections(image: np.ndarray, detections: Sequence[Detection], plates: Sequence[PlateResult]) -> np.ndarray:
    vis = image.copy()
    colors = {
        0: (0, 220, 0),
        1: (255, 160, 0),
        2: (0, 180, 255),
        3: (0, 0, 255),
    }
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.box]
        color = colors.get(det.class_id, (220, 220, 220))
        label = f"{YOLO_CLASSES[det.class_id]} {det.score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    for plate in plates:
        x1, y1, _, _ = plate.box
        status = "" if plate.valid else " INVALID"
        label = f"{plate.plate_text or 'INVALID'} {plate.lpr_score:.2f} [{plate.plate_type}]{status}"
        cv2.putText(vis, label, (x1, max(42, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2, cv2.LINE_AA)
    return vis


def load_rknn(path: str, core_mask: Optional[int]) -> RKNNLite:
    if RKNNLite is None:
        raise RuntimeError(f"rknnlite import failed: {RKNN_IMPORT_ERROR}")
    model = RKNNLite(verbose=False)
    ret = model.load_rknn(path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {path}")
    if core_mask is None:
        ret = model.init_runtime()
    else:
        ret = model.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {path}")
    return model


def parse_core_mask(name: str) -> Optional[int]:
    if RKNNLite is None or name == "default":
        return None
    mapping = {
        "core0": RKNNLite.NPU_CORE_0,
        "core1": RKNNLite.NPU_CORE_1,
        "core2": RKNNLite.NPU_CORE_2,
        "core01": RKNNLite.NPU_CORE_0_1,
        "core012": RKNNLite.NPU_CORE_0_1_2,
        "auto": RKNNLite.NPU_CORE_AUTO,
    }
    return mapping[name]


def write_debug_rows(debug: Optional[DebugWriter], image_path: Path, gt_text: Optional[str],
                     frame_shape: Tuple[int, int], detections: Sequence[Detection],
                     plates: Sequence[PlateResult], candidates_by_plate: Dict[int, List[Dict]],
                     timings: Dict[str, float], vis_path: Optional[str]) -> None:
    if debug is None:
        return

    image = str(image_path)
    height, width = frame_shape
    valid_plates = [p for p in plates if p.valid]
    best_plate = max(valid_plates, key=lambda p: p.lpr_score, default=None)
    debug.image_writer.writerow({
        "image": image,
        "gt_text": gt_text or "",
        "width": width,
        "height": height,
        "detection_count": len(detections),
        "plate_count": len(plates),
        "plate_texts": "|".join(p.plate_text for p in plates if p.valid),
        "best_plate_text": best_plate.plate_text if best_plate and best_plate.valid else "",
        "best_lpr_score": best_plate.lpr_score if best_plate else "",
        "best_det_score": best_plate.det_score if best_plate else "",
        "yolo_pre_ms": timings.get("yolo_pre_ms", 0.0),
        "yolo_infer_ms": timings.get("yolo_infer_ms", 0.0),
        "yolo_post_ms": timings.get("yolo_post_ms", 0.0),
        "lpr_total_ms": timings.get("lpr_total_ms", 0.0),
        "total_ms": timings.get("total_ms", 0.0),
        "vis_path": vis_path or "",
    })

    for det_idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det.box]
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        debug.det_writer.writerow({
            "image": image,
            "detection_index": det_idx,
            "class_id": det.class_id,
            "class_name": YOLO_CLASSES[det.class_id],
            "score": det.score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": w,
            "height": h,
            "area": w * h,
        })

    for plate_idx, plate in enumerate(plates):
        row = asdict(plate)
        x1, y1, x2, y2 = plate.box
        row.update({
            "image": image,
            "plate_index": plate_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
        debug.plate_writer.writerow(row)
        for candidate in candidates_by_plate.get(plate_idx, []):
            c_row = candidate.copy()
            c_row.update({"image": image, "plate_index": plate_idx})
            debug.candidate_writer.writerow(c_row)
    debug.flush()


def run_image(args, yolo, lpr, image_path: Path, output_dir: Path,
              debug: Optional[DebugWriter] = None) -> List[PlateResult]:
    total_start = time.perf_counter()
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"skip unreadable image: {image_path}")
        return []

    gt_text = parse_ccpd_gt(image_path)
    t0 = time.perf_counter()
    yolo_input, ratio, pad = preprocess_yolo(frame, args.yolo_color)
    yolo_pre_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    yolo_outputs = yolo.inference(inputs=[yolo_input])
    yolo_infer_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    detections = postprocess_yolo(yolo_outputs, ratio, pad, frame.shape[:2], args.conf_thres, args.nms_thres)
    yolo_post_ms = (time.perf_counter() - t0) * 1000.0

    crop_dir = output_dir / "crops"
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    plate_results: List[PlateResult] = []
    candidates_by_plate: Dict[int, List[Dict]] = {}
    lpr_total_ms = 0.0
    for det_idx, det in enumerate(detections):
        if det.class_id != 0:
            continue
        crop = crop_with_padding(frame, det.box, args.crop_pad_x, args.crop_pad_y)
        if crop is None:
            continue
        crop_h, crop_w = crop.shape[:2]
        estimated_type = estimate_plate_type(crop)
        lpr_input = preprocess_lpr(crop, args.lpr_color)

        t0 = time.perf_counter()
        logits = lpr.inference(inputs=[lpr_input])[0]
        raw_text, _ = greedy_debug_decode(logits)
        text, lpr_score, beam_score, decoded_type, subtype, candidates = decode_with_type_fallback(
            logits, estimated_type, args.beam_width, args.beam_topk
        )
        lpr_total_ms += (time.perf_counter() - t0) * 1000.0

        selected_candidate = next((c for c in candidates if c.get("selected")), None)
        valid = bool(text) and bool(selected_candidate and selected_candidate.get("length_ok"))
        invalid_reason = "" if valid else "invalid_length_or_empty"
        if not valid:
            lpr_score = 0.0

        plate_type = decoded_type if not subtype else f"{decoded_type}:{subtype}"
        crop_path = None
        if args.save_crops:
            crop_path = str(crop_dir / f"{image_path.stem}_plate{len(plate_results)}.jpg")
            cv2.imwrite(crop_path, crop)

        match = None if gt_text is None or not valid else text == gt_text
        plate_results.append(PlateResult(
            box=[int(round(v)) for v in det.box],
            detection_index=det_idx,
            det_score=det.score,
            estimated_type=estimated_type,
            decoded_type=decoded_type,
            plate_subtype=subtype,
            plate_type=plate_type,
            plate_text=text,
            valid=valid,
            invalid_reason=invalid_reason,
            lpr_score=lpr_score,
            beam_score=beam_score,
            raw_text=raw_text,
            crop_path=crop_path,
            crop_width=crop_w,
            crop_height=crop_h,
            gt_text=gt_text,
            match=match,
        ))
        candidates_by_plate[len(plate_results) - 1] = candidates

    vis_path = None
    if args.save_vis:
        output_dir.mkdir(parents=True, exist_ok=True)
        vis = draw_detections(frame, detections, plate_results)
        vis_path = str(output_dir / f"{image_path.stem}_vis.jpg")
        cv2.imwrite(vis_path, vis)

    timings = {
        "yolo_pre_ms": yolo_pre_ms,
        "yolo_infer_ms": yolo_infer_ms,
        "yolo_post_ms": yolo_post_ms,
        "lpr_total_ms": lpr_total_ms,
        "total_ms": (time.perf_counter() - total_start) * 1000.0,
    }
    payload = {
        "image": str(image_path),
        "gt_text": gt_text,
        "timings": timings,
        "detections": [
            {"class": YOLO_CLASSES[d.class_id], "score": d.score, "box": [int(round(v)) for v in d.box]}
            for d in detections
        ],
        "plates": [asdict(plate) for plate in plate_results],
    }
    print(json.dumps(payload, ensure_ascii=False))
    if debug is not None:
        debug.write_jsonl(payload)
        write_debug_rows(debug, image_path, gt_text, frame.shape[:2], detections, plate_results,
                         candidates_by_plate, timings, vis_path)
    return plate_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RKNN YOLO + LPRNet plate inference with constrained CTC decode.")
    parser.add_argument("--image", required=True, help="Input image path or image folder.")
    parser.add_argument("--yolo-model", default="fenqusai/rknn/best.rknn")
    parser.add_argument("--lpr-model", default="fenqusai/rknn/lprnet_unified_p15_focus_fp.rknn")
    parser.add_argument("--output-dir", default="fenqusai/tools/pipeline/result_rknn_plate")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-thres", type=float, default=0.45)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--beam-topk", type=int, default=8)
    parser.add_argument("--crop-pad-x", type=float, default=0.08)
    parser.add_argument("--crop-pad-y", type=float, default=0.15)
    parser.add_argument("--yolo-color", choices=["rgb", "bgr"], default="rgb",
                        help="Color order fed to YOLO RKNN after letterbox. Confirm against convert.py/export.")
    parser.add_argument("--lpr-color", choices=["bgr", "rgb"], default="bgr",
                        help="Color order fed to LPRNet RKNN. Current convert.py expects uint8 BGR crop when mean/std is in RKNN.")
    parser.add_argument("--core-mask", choices=["default", "core0", "core1", "core2", "core01", "core012", "auto"],
                        default="default")
    parser.add_argument("--lpr-core-mask", choices=["default", "core0", "core1", "core2", "core01", "core012", "auto"],
                        default="default")
    parser.add_argument("--save-vis", action="store_true", default=True)
    parser.add_argument("--no-save-vis", dest="save_vis", action="store_false")
    parser.add_argument("--save-crops", action="store_true", default=True)
    parser.add_argument("--no-save-crops", dest="save_crops", action="store_false")
    parser.add_argument("--export-debug", action="store_true", default=True,
                        help="Write debug CSV/JSONL files under <output-dir>/debug.")
    parser.add_argument("--no-export-debug", dest="export_debug", action="store_false")
    args = parser.parse_args()

    if cv2 is None:
        raise RuntimeError(f"opencv-python import failed: {CV2_IMPORT_ERROR}")
    if np is None:
        raise RuntimeError(f"numpy import failed: {NP_IMPORT_ERROR}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug = DebugWriter(output_dir, args) if args.export_debug else None
    yolo = load_rknn(args.yolo_model, parse_core_mask(args.core_mask))
    lpr = load_rknn(args.lpr_model, parse_core_mask(args.lpr_core_mask))
    try:
        for img_path in image_files(Path(args.image)):
            run_image(args, yolo, lpr, img_path, output_dir, debug)
    finally:
        yolo.release()
        lpr.release()
        if debug is not None:
            debug.close()


if __name__ == "__main__":
    main()

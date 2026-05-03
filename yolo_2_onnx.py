from ultralytics import YOLO

model = YOLO(r'C:\\Users\\Y9000P\\Downloads\\2026ICContest\\ICcontest_project\\runs\\new_plate_detect_merged\\weights\\best.pt')

model.export(
    format='onnx',
    opset=12,
    simplify=True,
    dynamic=False,      # ? 关闭动态输入（输入尺寸固定为640x640）
    nms=False           # ? 启用 NMS
)

# NMS（Non-Maximum Suppression，非极大值抑制）
# ? 用来 “去掉重复框，只保留最靠谱的那个框”
# 默认关闭了 NMS，输出的 ONNX 模型中不包含 NMS 的计算图，这样在推理时就需要单独实现 NMS 了

# 关闭动态输入（dynamic=False）后，导出的 ONNX 模型将固定输入尺寸为 640x640，这样在推理时就不需要进行动态调整了
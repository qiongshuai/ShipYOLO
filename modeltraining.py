from ultralytics import YOLO

# 加载预模型
model = YOLO('yolov10s.pt')

# 训练模型
model.train(
    data='data.yaml',
    epochs=5,
    imgsz=512,
    batch=16,
    augment=True,
    plots=True,
)

print("模型训练完毕！")

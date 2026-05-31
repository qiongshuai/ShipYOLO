from ultralytics import YOLO

# 指向具体的模型文件
weights_path = "D:/pycharm/PycharmProjects/boat/runs/detect/train3/weights/last.pt"

# 加载训练好的模型
model = YOLO(weights_path)

# 对图片进行推理
results = model.predict(source=r'D:/pycharm/PycharmProjects/boat/data/test/images',
                        show=True, save=True)

# # 打印推理结果
# for result in results:
#     print(result)

from ultralytics import YOLO

# Load a model
#model = YOLO('yolo11_SCCA.yaml')
#model = YOLO('D:/Pycharm/ultralytics-main_v11/ultralytics/cfg/models/11/yolo11m.yaml')
model = YOLO('D:/Pycharm/ultralytics-main_v11/ultralytics/cfg/models/v8/yolov8m.yaml')

# Train the model
results = model.train(data="D:/Pycharm/ultralytics-main_v11/datasets/chang'e_6/chang'e_6.yaml",
                      workers=0,
                      epochs=300,
                      batch=16,
                      imgsz=512)

# 获取训练的输出目录
train_dir = results.save_dir  # 训练结果的保存目录，例如 'runs/detect/train'

# 获取最后的性能参数
metrics = model.metrics  # 训练结束后的验证指标

# 提取需要的指标
final_precision = metrics.results_dict['metrics/precision(B)']
final_recall = metrics.results_dict['metrics/recall(B)']
final_map50 = metrics.results_dict['metrics/mAP50(B)']
final_map5095 = metrics.results_dict['metrics/mAP50-95(B)']

# 获取训练的输出目录
metrics_file = f"{train_dir}/train_metrics.txt"  # 动态生成文件路径
with open(metrics_file, "w") as f:
    f.write(f"Precision: {final_precision}\n")
    f.write(f"Recall: {final_recall}\n")
    f.write(f"mAP@0.5: {final_map50}\n")
    f.write(f"mAP@0.5:0.95: {final_map5095}\n")

from ultralytics import YOLO

model = YOLO(r'D:\Pycharm\ultralytics-main_v11\crater\runs\detect\train6\weights\best.pt')

source = r'D:\Pycharm\ultralytics-main_v11\datasets\changE6\images\ablation' #修改为自己的图片路径及文件名

# 运行推理，并附加参数
model.predict(source, save=True, save_txt=True, conf=0.3)
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("D:/Pycharm/ultralytics-main_v11/cater/runs/detect/train9/weights/best.pt")

    # 指定数据集配置文件
    data = "D:/Pycharm/ultralytics-main_v11/datasets/chang'e_4/320/chang'e_4_320.yaml"

    # 运行验证，针对测试集
    results = model.val(data=data, split='test')

    # 打印评价指标
    print("mAP@0.5:", results.box.map50)  # mAP@0.5
    print("mAP@0.5:0.95:", results.box.map)  # mAP@0.5:0.95
    # print("精度:", results.box.p)  # 精度
    # print("召回率:", results.box.r)  # 召回率

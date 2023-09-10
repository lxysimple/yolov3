# coding=utf-8
# project

# 数据集所在路径、项目所在路径
# 可自定义
DATA_PATH = "E:\github\yolov3\data\VOC"
PROJECT_PATH = "E:\github\yolov3"

# 目标检测的具体类别
# 可自定义
DATA = {"CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'],
        "NUM":20}

# model
# 不同scale下框的大小，分别是13×13，26×26，52×52特征图上框的大小
# 不建议自定义
# 这里的小目标值的是特征图分辨率很高是52*52
MODEL = {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj
         "STRIDES":[8, 16, 32], # 步长，缩放倍率
         "ANCHORS_PER_SCLAE":3 # 每个像素点产生3个框
         }

# train
TRAIN = {
         "TRAIN_IMG_SIZE":448, # 训练数据尺寸，可自定义
         "AUGMENT":True, # 训练时是否做数据扩增操作
         "BATCH_SIZE":1, # 初始batchsize=4
         "MULTI_SCALE_TRAIN":True, # 用同一图片不同尺寸进行训练
         "IOU_THRESHOLD_LOSS":0.5, 
         "EPOCHS":50,
         "NUMBER_WORKERS":1, # 用4个进程加载数据

         # 以下是和学习率迭代要使用的参数
         # 可自定义
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":1e-4,
         "LR_END":1e-6,
         "WARMUP_EPOCHS":2  # 预热，学习率一开始不要达到1e-4，而是慢慢到这个数
         }


# test
TEST = {
        "TEST_IMG_SIZE":448, # 测试数据尺寸
        "BATCH_SIZE":4,
        "NUMBER_WORKERS":2,
        "CONF_THRESH":0.01,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False
        }

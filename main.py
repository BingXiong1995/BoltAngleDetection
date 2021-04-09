import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
import sys
import utlis
import random

# 设置项目目录
sys.path.append("..")

# 导入工具类
from utils import label_map_util
from utils import visualization_utils as vis_util

# 存储模型的文件夹
MODEL_NAME = 'inference_graph'
# 需要检测的图片
IMAGE_NAME = '50.jpg'

# 获取当前路劲
CWD_PATH = os.getcwd()

# .pb文件的路径
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# label的路劲
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

# 检测图片路径拼接
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# 一共有多少个CLASS
NUM_CLASSES = 1

# 导入label以及对应的class
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 将模型加入到内存
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# 初始化tensor
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# 读取图片 [1, None, None, 3]
image = cv2.imread(PATH_TO_IMAGE)

# resize
scale_percent = 20
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# 增加一个维度 因为原本的网络是多个图片一起输入
image_expanded = np.expand_dims(image, axis=0)

# 进行检测
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# 对图像进行深拷贝
img_copy = image.copy()

# 画出检测的结果
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.8)

# 识别出之后拿到多个螺栓 + 标记的区域 将这些区域进行截取
height, width = img_copy.shape[:2]
box = np.squeeze(boxes)
max_boxes_to_draw = box.shape[0]
scores = np.squeeze(scores)
min_score_thresh = 0.8
for i in range(min(max_boxes_to_draw, box.shape[0])):
    if scores[i] > min_score_thresh:
        print(str(i))
        ymin = (int(box[i, 0] * height))
        xmin = (int(box[i, 1] * width))
        ymax = (int(box[i, 2] * height))
        xmax = (int(box[i, 3] * width))
        print(xmin, ymin, xmax, ymax)
        # 这里加一个padding
        bolt_roi = img_copy[ymin-10:ymax+10, xmin-10:xmax+10]
        # cv2.imshow("bolt_roi", bolt_roi)
        cv2.imwrite("./bolt_roi/" + str(i) + ".jpg", bolt_roi)
        # 开始检测角度 cThr参数需要根据不同的工况调节！
        imgContours, conts = utlis.getContours(bolt_roi, cThr=[0, 150], minArea=500, filter=4, showCanny=False,
                                               draw=False)
        if len(conts) != 0:
            biggest = conts[0][2]  # 最大区域的四个点
            # 对方形区域进行仿射变换使视野变换到垂直向下
            wP = 400  # 宽度
            hP = 400  # 高度
            imgWarp = utlis.warpImg(bolt_roi, biggest, wP, hP, pad=150)
            angle = utlis.findLine(imgWarp, show_edges=False, show_results=False, hough_thresh=140)
            print("angle", angle)
            cv2.putText(image, 'Angle:' + str(int(angle)), (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (255, 0, 0), 2)

# resize image
# scale_percent = 20
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dim = (width, height)
# resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('Bolt Angle Detector', image)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()

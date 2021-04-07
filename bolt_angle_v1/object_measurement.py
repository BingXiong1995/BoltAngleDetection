import cv2
import utlis

# 设置摄像头
webcam = False # True则直接从摄像头采集 False则直接从图片进行读取
# path = '../bolt_roi/5.jpg' # 图片的文件名 如果是读取图片的话
path = "../bolt_roi/11.jpg"
cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
cap.set(10, 160)  # 设置亮度
cap.set(3, 1920)  # 设置宽度
cap.set(4, 1080)  # 设置高度
scale = 2
wP = 200 * scale  # 宽度1
hP = 200 * scale  # 高度
###################################

cv2.namedWindow("Config")
cv2.createTrackbar("low", "Config", 0, 255, lambda x: None)
cv2.createTrackbar("high", "Config", 0, 255, lambda x: None)
cv2.createTrackbar("hough", "Config", 100, 255, lambda x: None)

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    # 获取到四边形的轮廓 conts为轮廓的部分
    low = cv2.getTrackbarPos("low", "Config")
    high = cv2.getTrackbarPos("high", "Config")
    hough = cv2.getTrackbarPos("hough", "Config")

    if high < low: high = low

    imgContours, conts = utlis.getContours(img, cThr=[low, high], minArea=500, filter=4, showCanny=True, draw=True)

    # 如果监测到了方形的轮廓就对螺钉进行进一步处理
    if len(conts) != 0:
        biggest = conts[0][2] # 最大区域的四个点
        # print(biggest)
        # 对方形区域进行仿射变换使视野变换到垂直向下
        imgWarp = utlis.warpImg(img, biggest, wP, hP, pad=80)
        utlis.findLine(imgWarp, show_edges=True, show_results=True, hough_thresh=hough)
        cv2.imshow("PerspectiveTransform", imgWarp)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_raw = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Original', resized_raw)
    cv2.waitKey(1)
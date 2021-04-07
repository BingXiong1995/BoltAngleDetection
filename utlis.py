import cv2
import numpy as np
import math
import time


'''
检测轮廓
params:
    img 原图
    cThr 轮廓检测的阈值
    showCanny 是否显示计算轮廓的结果
    minArea 最小区域
    filter 矩形的点数
    draw 是否画出轮廓区域
'''


def getContours(img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    # 键图像边缘化
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    # cv2.imshow('imgGray', cv2.resize(imgGray, (200, 200)))

    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 5)  # 高斯平滑 这一步是为了在下一部找边缘的时候去除一些噪音 因为边缘对噪声敏感
    # cv2.imshow('imgBlur', cv2.resize(imgBlur, (200, 200)))

    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1], True)  # 非微分边缘检测算子 有两个参数 第一个是边缘的阈值 第二个参数大可以检测较为明显的阈值, 最后一个参数是L2范数
    # 闭运算
    kernel = np.ones((5, 5))  # 定义一个用于闭运算
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)  # 膨胀运算
    imgThre = cv2.erode(imgDial, kernel, iterations=2)  # 腐蚀运算
    if showCanny:
        scale_percent = 20  # percent of original size
        width = int(imgThre.shape[1] * scale_percent / 100)
        height = int(imgThre.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_imgThre = cv2.resize(imgThre, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow('Canny', resized_imgThre)
        # cv2.imshow('Canny', cv2.resize(imgThre, (500, 800)))

    # 对轮廓进行查找
    # contours：返回的轮廓 hierarchy：每条轮廓对应的属性 轮廓的模式cv2.RETR_EXTERNAL只检测外轮廓
    # 轮廓的近似方法cv2.CHAIN_APPROX_SIMPLE压缩水平方向、垂直方向、对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需要4个点来保存轮廓信息
    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    # 找到最大的矩形轮廓
    for i in contours:
        area = cv2.contourArea(i)  # 计算轮廓的面积
        if area > minArea:  # 忽略掉小的轮廓
            peri = cv2.arcLength(i, True)  # 计算轮廓的周长
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # 得到的矩形实际上会认为是一条曲线 将这个曲线变为折线 返回折线的顶点集
            bbox = cv2.boundingRect(approx)  # 用一个最小矩形把找到的这个区域包裹起来
            if filter > 0:
                if len(approx) == filter:  # 当点集数等于4的时候我们认为找到了矩形 将矩形坐标记录下来
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    # 将找到的矩形区域从大到小排序
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            # 绘制出到找的轮廓 con是轮廓本身是一个list 以填充模式绘制
            cv2.drawContours(img, con[4], cv2.FILLED, (0, 0, 255), 5)
    return img, finalCountours


'''
得到的四个点顺序是不固定的 这会导致仿射变换的时候出现错误 因此我们找出上下左右四个点按照顺序排好
'''
def reorder(myPoints):
    # print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


'''
进行仿射变换
    img 输入的矩形区域的原图
    points 四个点
    w h 最终要投影出的矩形的宽高
    pad 打一个padding 直接去掉边缘部分的投影
'''
def warpImg(img, points, w, h, pad=20):
    # print(points)
    # 按照上下左右重新排序四个点
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # 计算透视变换矩阵 将图像投影到一个新的视平面 这个时候新的视平面并不是一个矩形
    # dst(x,y) = src((M11x+M12y+M13)/(M31x+M32y+M33), (M21x+M22y+M23)/(M31x+M32y+M33))
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    # 将透视变换的图像减去padding 防止把边缘也减进来了
    # cv2.imshow('imgWarp', cv2.resize(imgWarp,(200,200)))
    # cv2.imshow("temp", cv2.resize(imgDial,(500,800)))
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    # cv2.imshow('imgWarp-padding', cv2.resize(imgWarp, (200, 200)))

    return imgWarp


def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5


'''
霍夫直线检测 寻找直线上的line
    img 仿射变换后的bgr图像
重点在于多次调节阈值 只找到唯一一条线的时候返回角度
'''
def findLine(img, show_edges=False, show_results=False, hough_thresh = 118):
    init_thresh = 200
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    # cv2.imshow("edges", cv2.resize(edges, (200, 200)))
    while init_thresh > 20:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, init_thresh)  # 这里对最后一个参数使用了经验型的值
        # 如果检测到了就跳出循环
        if lines is not None:
            print("lines", lines.shape[0])
            break
        # 如果没有检测到就降低阈值继续检测
        else:
            init_thresh = init_thresh - 2
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)  # 这里对最后一个参数使用了经验型的值
    result = img.copy()
    # print(lines)
    # print("=============")
    avg_angel = 0
    count_line = 0
    try:
        for line in lines:
            count_line += 1
            rho = line[0][0]  # 第一个元素是距离rho
            theta = line[0][1]  # 第二个元素是角度theta 是弧度
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
                # 该直线与最后一行的焦点
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                cv2.line(result, pt1, pt2, (255), 2)  # 绘制一条白线
                radian = math.atan(np.abs(pt1[1] - pt2[1]) / np.abs(pt1[0] - pt2[0]))
                # print("angel:", str(int((radian / math.pi) * 180)))
                avg_angel += int((radian / math.pi) * 180)
            else:  # 水平直线
                pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
                # 该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                cv2.line(result, pt1, pt2, (255), 2)  # 绘制一条直线
                radian = math.atan(np.abs(pt1[1] - pt2[1]) / np.abs(pt1[0] - pt2[0]))

                avg_angel += int((radian / math.pi) * 180)
    except:
        print("No lines")

    try:
        avg_angel = avg_angel / count_line
        # cv2.imshow('Canny', edges)
        cv2.putText(result, 'Angle:' + str(int(avg_angel)), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                    (255, 0, 0), 2)
        cv2.imwrite('./bolt_roi/' + str(time.time()) + ".jpg", result)
        return str(int(avg_angel))
    except:
        print("count line cannot be 0")

    return False

    # cv2.imshow('Result', result)
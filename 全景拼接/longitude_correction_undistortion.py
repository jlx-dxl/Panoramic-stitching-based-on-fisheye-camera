import cv2
import numpy as np
import math

def imageEffectiveAreaInterception(img,ifshow_R,ifoutput_img):

    # input:
    # img:鱼眼相机原图
    # ifshow_R:是否print（R）
    # ifoutput_img:是否输出结果图
    # output:
    # img_valid:结果图
    # R:有效区域半径

    # if ifshow_R == True:
    #     print('R =', R)

    # if ifoutput_img == True:
    #     return img_valid

    # if ifoutput_img == False:
    #     cv2.imshow('fisheye', img)
    #     cv2.imshow("fisheye_valid", img_valid)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     return R,left,right,top,bottom

    # 设置灰度阈值
    T = 40

    # 转换为灰度图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 提取原图大小
    rows, cols = img.shape[:2]
    print(rows, cols)

    # 从上向下扫描
    for i in range(0, rows, 1):
        for j in range(0, cols, 1):
            if img_gray[i, j] >= T:
                if img_gray[i + 1, j] >= T:
                    top = i
                    break
        else:
            continue
        break
    # print('top =', top)

    # 从下向上扫描
    for i in range(rows - 1, -1, -1):
        for j in range(0, cols, 1):
            if img_gray[i, j] >= T:
                if img_gray[i - 1, j] >= T:
                    bottom = i
                    break
        else:
            continue
        break
    # print('bottom =', bottom)

    # 从左向右扫描
    for j in range(0, cols, 1):
        for i in range(top, bottom, 1):
            if img_gray[i, j] >= T:
                if img_gray[i, j + 1] >= T:
                    left = j
                    break
        else:
            continue
        break
    # print('left =', left)

    # 从右向左扫描
    for j in range(cols - 1, -1, -1):
        for i in range(top, bottom, 1):
            if img_gray[i, j] >= T:
                if img_gray[i, j - 1] >= T:
                    right = j
                    break
        else:
            continue
        break
    # print('right =', right)

    # 计算有效区域半径
    R = max((bottom - top) / 2, (right - left) / 2)
    R = int(R)
    if ifshow_R == True:
        print('R =', R)

    # 提取有效区域
    bottom=top+2*(R)
    right=left+2*(R)
    img_valid=img[top:bottom, left:right]

    if ifoutput_img == True:
        return  img_valid

    if ifoutput_img ==False:
        img_valid = cv2.resize(img_valid, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("fisheye_valid", img_valid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return R,left,right,top,bottom

def R_left_right_top_bottom_get():
    s=[]
    R=np.array([0,0,0])
    left=np.array([0,0,0])
    right=np.array([0,0,0])
    top=np.array([0,0,0])
    bottom=np.array([0,0,0])
    for index in range(3):
        s.append(cv2.VideoCapture(index))
        s[index].set(3,1920)
        s[index].set(4,1080)
        n = 0
        while True:
            n += 1
            if s[index].isOpened():
                ret,frame=s[index].read()
                if n >= 20:
                    R[index],left[index],right[index],top[index],bottom[index]=imageEffectiveAreaInterception(frame,True,False)
                    break
    print("R:",R)
    np.save(r"D:/大四上/传感器/全景拼接/data/R.npy",R)
    np.save(r"D:/大四上/传感器/全景拼接/data/left.npy",left)
    np.save(r"D:/大四上/传感器/全景拼接/data/right.npy",right)
    np.save(r"D:/大四上/传感器/全景拼接/data/top.npy",top)
    np.save(r"D:/大四上/传感器/全景拼接/data/bottom.npy",bottom)

def fisheye_longitude_correction():
    R=np.load(r"D:/大四上/传感器/全景拼接/data/R.npy")
    number_of_camera=3
    mapx = []
    mapy = []
    for index in range(number_of_camera):
        mapx.append(np.zeros([2 * R[index], 2 * R[index]]).astype(np.float32))
        mapy.append(np.zeros([2 * R[index], 2 * R[index]]).astype(np.float32))

    mapx = np.array(mapx, dtype=object)
    mapy = np.array(mapy, dtype=object)

    for index in range(number_of_camera):
        for i in range(mapx[index].shape[0]):
            for j in range(mapy[index].shape[1]):
                mapx[index][i, j] = (j - R[index]) / R[index] * (R[index] ** 2 - (i - R[index]) ** 2) ** 0.5 + R[index]
                mapy[index][i, j] = i

    np.save(r"D:/大四上/传感器/全景拼接/data/mapx.npy", mapx)
    np.save(r"D:/大四上/传感器/全景拼接/data/mapy.npy", mapy)












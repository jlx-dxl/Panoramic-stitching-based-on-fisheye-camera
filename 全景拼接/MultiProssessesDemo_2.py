import glob
import biaoding1
import os
import cv2
from Stitcher import Stitcher
import numpy as np
import multiprocessing
from multiprocessing import Queue
import time

# stitcher = Stitcher()
q = Queue()

# DIM = []
# K = []
# D = []
CAMERA_COUNT = 3  # 摄像头个数
#
# for index in range(1, CAMERA_COUNT + 1):
#     print(index)
#     DIM.append(np.load(r"D:/大四上/传感器/全景拼接/data/DIM" + str(index) + ".npy"))
#     K.append(np.load(r"D:/大四上/传感器/全景拼接/data/K" + str(index) + ".npy"))
#     D.append(np.load(r"D:/大四上/传感器/全景拼接/data/D" + str(index) + ".npy"))

def video_read(id):
    camera_id = id
    # 摄像头1
    if camera_id == 0:
        cap = cv2.VideoCapture(0)

    # 摄像头2
    if camera_id == 1:
        cap = cv2.VideoCapture(1)

    # 摄像头3
    if camera_id == 2:
        cap = cv2.VideoCapture(2)

    # 获取每一个视频的尺寸
    width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(width, height)

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = frame[0:480, 80:560]
        # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
        isEmpty = q.empty()
        print(isEmpty)
        if isEmpty == True:
            print('队列'+str(id)+'中无数据！')
            # q.put(frame)
            time.sleep(0.001)

        else:
            print('队列'+str(id)+'中有数据！')
            # Frame = q.get(frame)
            time.sleep(0.002)
            # frameUp = np.hstack(frame, Frame)#左右合并
        # frameUp = np.vstack((frameLeftUp, frameRightUp))#上下合并

        cv2.imshow('camera' + str(id), frame)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    cap.release()

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    print("主进程开始启动！")

    for index in range(CAMERA_COUNT):
        print('摄像头的索引号是：', index)
        p = multiprocessing.Process(target=video_read, args=(index,))
        p.start()









import cv2
import biaoding1
import numpy as np
from Stitcher import Stitcher

stitcher=Stitcher()

DIM = []
K = []
D = []
CAMERA_COUNT = 3  # 摄像头个数

for index in range(1, CAMERA_COUNT + 1):
    DIM.append(np.load(r"D:/大四上/传感器/全景拼接/data/DIM" + str(index) + ".npy"))
    K.append(np.load(r"D:/大四上/传感器/全景拼接/data/K" + str(index) + ".npy"))
    D.append(np.load(r"D:/大四上/传感器/全景拼接/data/D" + str(index) + ".npy"))

s1 = cv2.VideoCapture(0)
s2 = cv2.VideoCapture(1)
s3 = cv2.VideoCapture(2)
s1.set(3,1920)
s1.set(4,1080)
s2.set(3,1920)
s2.set(4,1080)
s3.set(3,1920)
s3.set(4,1080)


H_list=[]
index=0
if s1.isOpened():
    if s2.isOpened():
        if s3.isOpened():
            while True:
                ret1,frame1=s1.read()
                frame1=frame1[0:1080,390:1470]
                frame1=biaoding1.undistort_frames(frame1, DIM[0], K[0], D[0])
                # frame1=frame1[60:360,60:360]

                ret2,frame2=s2.read()
                frame2 = frame2[0:1080,390:1470]
                frame2 = biaoding1.undistort_frames(frame2, DIM[1], K[1], D[1])
                # frame2=frame2[60:360,60:360]

                ret3,frame3=s3.read()
                frame3 = frame3[0:1080,390:1470]
                frame3 = biaoding1.undistort_frames(frame3, DIM[2], K[2], D[2])
                # frame3=frame3[60:360,60:360]

                index += 1

                if index>=20:
                    # (result_right, H, vis_right) = stitcher.stitch([frame2, frame3], showMatches=True)
                    # H_list.append(H)
                    # print(index-20,":",H_list[index-20])
                    # (result_right, vis_right) = stitcher.stitchleft([frame2, frame3], showMatches=True)
                    # result_right = cv2.resize(result_right, None, fx=0.5, fy=0.5)
                    # (result_left, vis_left) = stitcher.stitchleft([frame1, result_right], showMatches=True)
                    frame1 = cv2.resize(frame1, None, fx=0.6, fy=0.6)
                    frame2 = cv2.resize(frame2, None, fx=0.6, fy=0.6)
                    frame3 = cv2.resize(frame3, None, fx=0.6, fy=0.6)

                    cv2.imshow("camera1", frame1)
                    cv2.imshow("camera2", frame2)
                    cv2.imshow("camera3", frame3)
                    # cv2.imshow("result_right", result_right)
                    # cv2.imshow("vis_right",vis_right)
                    # cv2.imshow("result",result_left)
                if cv2.waitKey(25) & 0xff==(ord('q')):
                    break

s1.release()
s2.release()
s3.release()
cv2.destroyAllWindows()





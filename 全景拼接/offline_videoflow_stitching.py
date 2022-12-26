import cv2
import numpy as np
from Stitcher import Stitcher

s=[]
number_of_camera=3
for index in range(number_of_camera):
    s.append(cv2.VideoCapture(index))
    s[index].set(3,1920)
    s[index].set(4,1080)

R=np.load(r"D:/大四上/传感器/全景拼接/data/R.npy")
left=np.load(r"D:/大四上/传感器/全景拼接/data/left.npy")
right=np.load(r"D:/大四上/传感器/全景拼接/data/right.npy")
top=np.load(r"D:/大四上/传感器/全景拼接/data/top.npy")
bottom=np.load(r"D:/大四上/传感器/全景拼接/data/bottom.npy")
mapx = np.load(r"D:/大四上/传感器/全景拼接/data/mapx.npy",allow_pickle=True)
mapy = np.load(r"D:/大四上/传感器/全景拼接/data/mapy.npy",allow_pickle=True)

H_right=np.load(r"D:/大四上/传感器/全景拼接/data/H/H_right53.npy")
H_left=np.load(r"D:/大四上/传感器/全景拼接/data/H/H_left62.npy")

stitcher=Stitcher()
frame=[None,None,None,None]
n=0
w=960
h=960

while True:
    n += 1
    if n <= 10:
      print("n=",n)
    for index in range(number_of_camera):

        ret1,frame[index]=s[index].read()
        frame[index]=frame[index][top[index]:bottom[index], left[index]:right[index]]
        frame[index] = cv2.remap(frame[index], mapx[index], mapy[index], interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    frame[0]=frame[0][int(0.5*(frame[0].shape[0]-h)):int(0.5*(frame[0].shape[0]+h)),int(0.5*(frame[0].shape[1]-w)):int(0.5*(frame[0].shape[1]+w))]
    frame[1]=frame[1][int(0.5*(frame[1].shape[0]-h)):int(0.5*(frame[1].shape[0]+h)),int(0.5*(frame[1].shape[1]-w)):int(0.5*(frame[1].shape[1]+w))]
    frame[2]=frame[2][int(0.5*(frame[2].shape[0]-h)):int(0.5*(frame[2].shape[0]+h)),int(0.5*(frame[2].shape[1]-w)):int(0.5*(frame[2].shape[1]+w))]

    if n>=10:
        if n==10:
            print("strat video")
        result_right = stitcher.offlinestitch_right([frame[1], frame[0]],H_right)
        result_left= stitcher.offlinestitch_left([frame[0], frame[2]],H_left)
        result=stitcher.stitch_all([result_left,frame[0],result_right])

        # result_right = cv2.resize(result_right, None, fx=0.5, fy=0.5)
        # result_left = cv2.resize(result_left, None, fx=0.5, fy=0.5)
        result = cv2.resize(result, None, fx=0.55, fy=0.55)


        # for index in range(1, number_of_camera+1):
        #     frame[index] = cv2.resize(frame[index], (0, 0), fx=0.5, fy=0.5)
        #     if n == 10:
        #         print("the highth of camera" + str(index) + " is:", frame[index].shape[0],
        #               "the width of camera" + str(index) + " is:", frame[index].shape[1])
        #     cv2.imshow("camera"+str(index),frame[index])

        # cv2.imshow("result_right", result_right)
        # cv2.imshow("result_left", result_left)
        cv2.imshow("result", result)

    a=cv2.waitKey(25)
    if a == (ord('q')):
        break

for index in range(number_of_camera):
    s[index].release()
cv2.destroyAllWindows()
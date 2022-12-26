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

stitcher=Stitcher()
frame=[None,None,None,None]
n=0
w=960
h=960
# H_R=[]
# H_L=[]
i = 0
j = 0

while True:
    n += 1
    if n <= 10:
      print("n=",n)
    for index in range(number_of_camera):

        ret1,frame[index]=s[index].read()
        frame[index]=frame[index][top[index]:bottom[index], left[index]:right[index]]
        # frame[index] = cv2.remap(frame[index], mapx[index], mapy[index], interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    # frame[0]=frame[0][int(0.5*(frame[0].shape[0]-h)):int(0.5*(frame[0].shape[0]+h)),int(0.5*(frame[0].shape[1]-w)):int(0.5*(frame[0].shape[1]+w))]
    # frame[1]=frame[1][int(0.5*(frame[1].shape[0]-h)):int(0.5*(frame[1].shape[0]+h)),int(0.5*(frame[1].shape[1]-w)):int(0.5*(frame[1].shape[1]+w))]
    # frame[2]=frame[2][int(0.5*(frame[2].shape[0]-h)):int(0.5*(frame[2].shape[0]+h)),int(0.5*(frame[2].shape[1]-w)):int(0.5*(frame[2].shape[1]+w))]


    if n>=10:
        if n==10:
            print("strat video")
        # result_right,H_right = stitcher.stitchright([frame[1], frame[0]])
        # result_left,H_left = stitcher.stitchleft([frame[0], frame[2]])
        # result=stitcher.stitch_all([result_left,frame[0],result_right])
        # # # result_put=stitcher.put_stitch([frame[1], frame[0],frame[2]])
        # np.save(r"D:/大四上/传感器/全景拼接/data/H/H_right" + str(i) + ".npy", H_right)
        # print("i=",i)
        # i += 1
        # np.save(r"D:/大四上/传感器/全景拼接/data/H/H_left"+str(j)+".npy", H_left)
        # print("j=",j)
        # j +=1
        #
        # result_right = cv2.resize(result_right, None, fx=0.3, fy=0.3)
        # result_left = cv2.resize(result_left, None, fx=0.3, fy=0.3)
        # result = cv2.resize(result, None, fx=0.3, fy=0.3)
        # result_put = cv2.resize(result_put, None, fx=0.5, fy=0.5)

        for index in range(number_of_camera):
            # if n == 10:
                # print("the highth of camera" + str(index) + " is:", frame[index].shape[0],
                #       "the width of camera" + str(index) + " is:", frame[index].shape[1])
            frame[index] = cv2.resize(frame[index], (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("camera"+str(index),frame[index])

        # cv2.imshow("result_right", result_right)
        # cv2.imshow("result_left", result_left)
        # cv2.imshow("result", result)
        # cv2.imshow("result_put", result_put)

    a=cv2.waitKey(25)
    # if a == (ord('d')):
    #     np.save(r"D:/大四上/传感器/全景拼接/data/H/H_right"+str(i)+".npy", H_right)
    #     print("H_right successfully saved!!!")
    #     i +=1
    # if a == (ord('a')):
    #     np.save(r"D:/大四上/传感器/全景拼接/data/H/H_left"+str(j)+".npy", H_left)
    #     print("H_left successfully saved!!!")
    #     j +=1
    if a == (ord('q')):
        break

for index in range(number_of_camera):
    s[index].release()
cv2.destroyAllWindows()




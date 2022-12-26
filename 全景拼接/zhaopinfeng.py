import cv2
import numpy as np
from Stitcher import Stitcher
import glob

left=np.load(r"D:/大四上/传感器/全景拼接/data/left.npy")
right=np.load(r"D:/大四上/传感器/全景拼接/data/right.npy")
top=np.load(r"D:/大四上/传感器/全景拼接/data/top.npy")
bottom=np.load(r"D:/大四上/传感器/全景拼接/data/bottom.npy")
mapx = np.load(r"D:/大四上/传感器/全景拼接/data/mapx.npy",allow_pickle=True)
mapy = np.load(r"D:/大四上/传感器/全景拼接/data/mapy.npy",allow_pickle=True)
root_path_1 = r'D:\imageStitching\12.17\camera1\1_10.jpg'
root_path_2 = r'D:\imageStitching\12.17\camera2\2_10.jpg'
root_path_3 = r'D:\imageStitching\12.17\camera3\3_10.jpg'
root_path=[None,root_path_1,root_path_2,root_path_3]

w=960
h=960
frame = [cv2.imread(img) for img in root_path]
for index in range(1,4):
    frame[index]=frame[index][top[index]:bottom[index], left[index]:right[index]]
    frame[index] = cv2.remap(frame[index], mapx[index], mapy[index], interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    frame[index]=frame[index][int(0.5*(frame[index].shape[0]-h)):int(0.5*(frame[index].shape[0]+h)),int(0.5*(frame[index].shape[1]-w)):int(0.5*(frame[index].shape[1]+w))]
    # cv2.imshow("frame"+str(index),frame[index])
    # cv2.waitKey(0)

sti = np.zeros((1200,1920,3), dtype=np.uint8)
for i in range(50):
        sti[8:960+8,0:960-150] = frame[1][0:960,0:960-150]
        sti[0:960,960-150:1920-150-78] = frame[3][0:960,78:960]
        cv2.imshow('result',sti)
        a = cv2.waitKey(0)
        if a == ord('s'):
            print(i)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (917,2473))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

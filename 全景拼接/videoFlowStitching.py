import numpy as np
import cv2 as cv
from Stitcher import Stitcher
import biaoding

# 读取视频
s1 = cv.VideoCapture(1)
s2 = cv.VideoCapture(2)
# s3 = cv2.VideoCapture(3)

# 读视频第一帧
success1, frame1 = s1.read()
while success1 is False:
    success1, frame1 = s1.read()
    print("success1 is false")
frame1 = frame1[0:480, 80:560]
success2, frame2 = s2.read()
while success2 is False:
    frame2 = frame2[0:480,80:560]
    print("success2 is false")
stitcher = Stitcher()
(result, vis) = stitcher.stitchleft([frame1, frame2], showMatches=True)
resultInfo = result.shape
size = (resultInfo[1],resultInfo[0])

# 获取帧率
fps1 = s1.get(cv.CAP_PROP_FPS)
fps2 = s2.get(cv.CAP_PROP_FPS)

# VideoWriter实例化
fourcc = cv.VideoWriter_fourcc('X','V','I','D')
writer = cv.VideoWriter('1.avi', fourcc , 5 , size )
result.dtype = np.uint8
writer.write(result)
i=0;j=0;

#摄像头标定
(ret1_1, mtx_1, dist_1, rvecs_1, tvecs_1)=biaoding.calibration(toBeCalibratedImages1,saveLocation_1,dstLocation_1)
(ret1_2, mtx_2, dist_2, rvecs_2, tvecs_2)=biaoding.calibration(toBeCalibratedImages2,saveLocation_2,dstLocation_2)

while success1:
    i=i+1
    success1, frame1 = s1.read()
    if frame1 is None:
        print("frame1 is none, break")
        break
    frame1=frame1[0:480,80:560]
    frame1=undistort(frame1,mtx_1, dist_1)
    print ("processing: ",i,",success1:",success1,"frame1",frame1[320,180,:])

    while success2:
        j=j+1
        success2, frame2 = s2.read()
        if frame2 is None:
            print("frame2 is none, break")
            break
        frame2=frame2[0:480,80:560]
        frame2=undistort(frame2,mtx_2, dist_2)
        print("processing: ", j, ",success2:", success2, "frame2", frame2[320, 180, :])

        (result, vis) = stitcher.stitchleft([frame1, frame2], showMatches=True)

        print("result:",result[320,180,:])
        result.dtype = np.uint8
        writer.write(result)

        cv.imshow("result", result);
        # if (cv.waitKey(3) & 0xff) == ord('q'):
        if (cv.waitKey(1) & 0xff) == 27:
            break
        break
    if (cv.waitKey(1) & 0xff) == 27:
        break

# 释放VideoCapture资源
s2.release()
s1.release()
# 释放VideoWriter资源
writer.release()
print("successfully released!")
cv.destroyAllWindows()


from Stitcher import Stitcher
import cv2
import numpy as np

# 读取视频
video1=cv2.VideoCapture('2_1.mp4')
video2=cv2.VideoCapture('2_2.mp4')

# 读视频第一帧
success1, frame1 = video1.read()
success2, frame2 = video2.read()
stitcher = Stitcher()
(result, vis) = stitcher.stitchleft([frame1, frame2], showMatches=True)
resultInfo = result.shape
size = (resultInfo[1],resultInfo[0])

# 获取帧率
fps1 = video1.get(cv2.CAP_PROP_FPS)
fps2 = video2.get(cv2.CAP_PROP_FPS)

# VideoWriter实例化
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
writer = cv2.VideoWriter('1.avi', fourcc , 5 , size )
result.dtype=np.uint8
writer.write(result)
i=0;j=0;

while success1:
    i=i+1
    success1, frame1 = video1.read()
    if frame1 is None:
        print("frame1 is none, break")
        break
    frame1=cv2.resize(frame1,(360,640))
    print ("processing: ",i,",success1:",success1,"frame1",frame1[320,180,:])

    while success2:
        j=j+1
        success2, frame2 = video2.read()
        if frame2 is None:
            print("frame2 is none, break")
            break
        frame2 = cv2.resize(frame2, (360, 640))
        print("processing: ", j, ",success2:", success2, "frame2", frame2[320, 180, :])

        (result, vis) = stitcher.stitchleft([frame1, frame2], showMatches=True)

        print("result:",result[320,180,:])
        result.dtype = np.uint8
        writer.write(result)

        cv2.imshow("result", result);
        # if (cv2.waitKey(3) & 0xff) == ord('q'):
        if (cv2.waitKey(1) & 0xff) == 27:
            break
        break
    if (cv2.waitKey(1) & 0xff) == 27:
        break

# 释放VideoCapture资源
video2.release()
video1.release()
# 释放VideoWriter资源
writer.release()
print("successfully released!")
cv2.destroyAllWindows()




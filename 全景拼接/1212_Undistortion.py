import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import shutil

def check_file_location(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

#设置超参数
CHECKKEYBOARD=(8,11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
radius = (11,11)

#获取特征点并做可视化
objpoints=[]
imgpoints=[]
objp=np.zeros((1,CHECKKEYBOARD[0]*CHECKKEYBOARD[1],3),np.float32)
objp[0,:,:2]=np.mgrid[:CHECKKEYBOARD[0],:CHECKKEYBOARD[1]].T.reshape(-1,2)

#获取标定图片集
toBeCalibratedImages=[]
saveLocation=[]
date="12.12"
numberOfCameras=3
for index in range(1,numberOfCameras+1):
    toBeCalibratedImages.append(sorted(glob.glob(r"D:/imageStitching/tobeCalibratedImages/camera"+str(index)+r"/"+r"*.jpg"),key=os.path.getmtime))

#保存地址
for index in range(1,numberOfCameras+1):
    saveLocation.append(r"D:/imageStitching/"+date+r"/undistortedImages/camera"+str(index)+r"/")
    if not os.path.exists(saveLocation[index-1]):
        os.makedirs(saveLocation[index-1])

mapx=[]
mapy=[]

#获取特征点并做可视化
for index in range(numberOfCameras):
    for path in tqdm(toBeCalibratedImages[index]):
        image=cv2.imread(path)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,CHECKKEYBOARD,None)
        if ret == True:
            objpoints.append(objp)
            corners=cv2.cornerSubPix(gray,corners,radius,(-1,-1),criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(image,CHECKKEYBOARD,corners,True)

    #计算相机参数矩阵K和畸变矩阵D
    h,w=gray.shape[:2]
    flags=0
    flags |=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    flags |=cv2.fisheye.CALIB_CHECK_COND
    flags |=cv2.fisheye.CALIB_FIX_SKEW
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,1e-6)
    ret,K,D,rvecs,tvecs=cv2.fisheye.calibrate(objpoints,imgpoints,(w,h),K=None,D=None,rvecs=None,tvecs=None,flags=flags,criteria=criteria)
    print(index,"  ",ret)

    #计算参数矩阵并保存
    map_combine,_=cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),K,(w,h),cv2.CV_16SC2)
    mapx.append(map_combine[:,:,0].astype(np.float32))
    mapy.append(map_combine[:,:,1].astype(np.float32))
    check_file_location("./npy/")
    np.save('./npy/mapy.npy',mapy)
    np.save('./npy/mapx.npy',mapx)

#载入参数矩阵和鱼眼原图，并处理
mapx=np.load('./npy/mapx.npy')
mapy=np.load('./npy/mapy.npy')

for index in range(numberOfCameras):
    for path in tqdm(toBeCalibratedImages[index]):
        file_name = os.path.basename(path)
        image = cv2.imread(path)
        image_remap = cv2.remap(image,mapx[index],mapy[index],interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(saveLocation[index]+file_name+'.jpg', image_remap)







import glob
import numpy as np
import biaoding1
import os

# toBeCalibratedImages1 = sorted(glob.glob(r"D:/imageStitching/tobeCalibratedImages_1213/camera1/*.jpg"),key=os.path.getmtime)
# toBeCalibratedImages2 = sorted(glob.glob(r"D:/imageStitching/tobeCalibratedImages_1213/camera2/*.jpg"),key=os.path.getmtime)
toBeCalibratedImages3 = sorted(glob.glob(r"D:/imageStitching/tobeCalibratedImages_1213/camera3/*.jpg"),key=os.path.getmtime)
saveLocation = r"D:/imageStitching/12.13/camera3/"

#摄像头标定
# (ret1_2, mtx_2, dist_2, rvecs_2, tvecs_2)=biaoding1.calibration(toBeCalibratedImages1,saveLocation)
# DIM1,K1,D1=biaoding1.calibration(toBeCalibratedImages1)
# np.save(r"D:/大四上/传感器/全景拼接/data/DIM1.npy",DIM1)
# np.save(r"D:/大四上/传感器/全景拼接/data/K1.npy",K1)
# np.save(r"D:/大四上/传感器/全景拼接/data/D1.npy",D1)
# DIM2,K2,D2=biaoding1.calibration(toBeCalibratedImages2)
# np.save(r"D:/大四上/传感器/全景拼接/data/DIM2.npy",DIM2)
# np.save(r"D:/大四上/传感器/全景拼接/data/K2.npy",K2)
# np.save(r"D:/大四上/传感器/全景拼接/data/D2.npy",D2)
DIM3,K3,D3=biaoding1.calibration(toBeCalibratedImages3)
np.save(r"D:/大四上/传感器/全景拼接/data/DIM3.npy",DIM3)
np.save(r"D:/大四上/传感器/全景拼接/data/K3.npy",K3)
np.save(r"D:/大四上/传感器/全景拼接/data/D3.npy",D3)

for fname in toBeCalibratedImages3:
    undistorted_img=biaoding1.undistort_Images(fname,DIM3,K3,D3,saveLocation)


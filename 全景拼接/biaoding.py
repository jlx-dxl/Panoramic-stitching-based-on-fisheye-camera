# -*- coding: utf-8 -*-
"""
Created at Thu Dec 7 16:43:40 2021

@author: JLX
"""

import numpy as np
import cv2 as cv
# import glob
import os

# toBeCalibratedImages_1 = sorted(glob.glob(r"D:/imageStitching/DATA/camera1/*.jpg"),key=os.path.getmtime)
# saveLocation_1 = r"D:/imageStitching/DATA/camera1_calibrated/"
#
# toBeCalibratedImages_2 = glob.glob(r"D:/imageStitching/DATA/camera2/*.jpg",key=os.path.getmtime)
# saveLocation_2 = r"D:/imageStitching/DATA/camera2_calibrated/"
#
# toBeCalibratedImages_2 = glob.glob(r"D:/imageStitching/DATA/camera3/*.jpg",key=os.path.getmtime)
# saveLocation_2 = r"D:/imageStitching/DATA/camera3_calibrated/"


def calibration(toBeCalibratedImages, saveLocation):

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points
    objp = np.zeros((8 * 11, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)

    i = 0
    for fname in toBeCalibratedImages:
        i = i + 1
        file_name = os.path.basename(fname)
        img = cv.imread(fname)
        # print(fname.index)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
        # If found, add object points, image points (after refining them)
        if ret == False:
            print(file_name, "can not be calibrated",)
            continue
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (6, 6), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (11, 8), corners2, ret)
            cv.imshow('img', img)
            cv.imwrite(saveLocation + file_name + "_calibrated" + '.jpg', img)
            print(file_name," successfully calibrated")
            cv.waitKey(500)
        ret1, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort 1
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv.imshow('dst', dst)
        cv.imwrite(saveLocation + file_name + "_undistorted" + "with method1" + '.jpg', dst)
        print(file_name, " successfully undistorted")
        cv.waitKey(500)

        # # undistort 2
        # mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]
        # cv.imshow('dst', dst)
        # cv.imwrite(dstLocation + '2_' + str(i) + '.jpg', dst)
        # cv.waitKey(500)

    cv.destroyAllWindows()
    return (ret1, mtx, dist, rvecs, tvecs)

def undistort1 (img , mtx , dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

def undistort2 (img , mtx , dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst



# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error
# print( "total error: {}".format(mean_error/len(objpoints)) )

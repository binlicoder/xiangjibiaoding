#!/usr/bin/env python
#ref:https://learnopencv.com/camera-calibration-using-opencv/
# 原文链接：https://blog.csdn.net/qq_43528254/article/details/108276225
import cv2
import numpy as np
import os
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (5, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*1.5
prev_img_shape = None

# Extracting path of individual image stored in a given directory
# images = glob.glob(r'G:/Microsee/camera_calibration/CameraCalibration-master/src/images/calibrated/1/*.png')
images = glob.glob(r'./src/images/calibrated/1/*.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        cv2.imwrite("./images/"+fname[-8:], img)
    else:
        print(fname)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()

# h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret: \n")
print(ret)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)
# images to be calibrated
# datadir=r"G:/Microsee/camera_calibration/CameraCalibration-master/src/images/raw_ring/"
datadir = r"./src/images/raw_ring/"
path=os.path.join(datadir)
img_list=os.listdir(path)

for i in img_list:
    img=cv2.imread(os.path.join(path,i))
    h,w=img.shape[:2]
    newMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) #矫正图像

    dst = cv2.undistort(img, mtx, dist, None, newMatrix)
    #calibrated images stored here
    cv2.imwrite('.\\images\\calibrated\\'+i,dst)

#计算重投影误差
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

#输出参数
print("-------------calibrated----------------")
print('ret:\n',ret)
print('mtx:\n',mtx)
print('dist:\n',dist)
print('rvecs:\n',rvecs)
print('tvecs:\n',tvecs)
print ("total error: ", tot_error/len(objpoints))


#计算整批图像 每个像素对应多少个毫米

import cv2
import numpy as np
import os
import glob
import math

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
# images = glob.glob('./images/uncalibrated/*.png')   #未校正的图像
images = glob.glob('./images/calibrated/1/*.png')   #已校正图像
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
    else:
        print("no chessboardcorners found in image {}".format(fname))

lst=[]
lst1=[]
for i in range(len(imgpoints)):
    for j in range(len(imgpoints[i])):
        if j%5!=4:
           lst.append(np.linalg.norm(imgpoints[i][j]-imgpoints[i][j+1]))
           #  lst.append(imgpoints[i][j])
        if j<30:
            # lst1.append(imgpoints[i][j])
            lst1.append(np.linalg.norm(imgpoints[i][j]-imgpoints[i][j+5]))
# print(len(lst))  #644/28=23
# print(len(lst1))  #690/30=23
print(1.5/np.mean(np.array(lst)))
print(1.5/np.mean(np.array(lst1)))
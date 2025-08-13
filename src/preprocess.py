import os
import glob
import sys, argparse
import pprint
import numpy as np
import cv2


# DATA_DIR = r'G:\\Microsee\\camera_calibration\\CameraCalibration-master\\src\\images\\calibrated'  ## "../data/"
DATA_DIR = r'./src/images/calibrated'
save_dir = r"./save/save1/"
# save_dir=r"G:/Microsee/camera_calibration/images/save1/"
images = [each for each in glob.glob(DATA_DIR + "*.png")]
images = sorted(images)
for each in images:
    grayImage = cv2.imread(each, 0)
    dst = cv2.bilateralFilter(grayImage, 0, 50, 30)  # 双边滤波
    ret, thresh =cv2.threshold(dst, 45, 255,cv2.THRESH_BINARY)
    # cv2.imwrite(each[:-4]+'1'+'.png', thresh)
    # yield (each, thresh)
    medianblur = cv2.medianBlur(np.uint8(thresh), 5)
    cv2.imwrite(save_dir+each[-8:],medianblur)
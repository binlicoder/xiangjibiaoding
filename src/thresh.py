# out_dir=r"G:/Microsee/camera_calibration/images/"
out_dir = r"./src/images/"
from PIL import Image     #??
from pylab import *       #Matplotlib 中的 PyLab 接口包含很多方便用户创建图像的函数
import numpy as np
import os

import copy
from PIL import Image
from pylab import *
from scipy.cluster.vq import *
import cv2 as cv

def get_imlist(path):
     """ 返回目录中所有 JPG 图像的文件名列表 """
     return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

im_list=get_imlist(out_dir)
# print(im_list[5])
# radius=3
# print(len(im_list))

for idx in im_list:
    # im=np.array(Image.open(idx))
    im=cv.imread(idx)
    # steps = 1080 # image is divided in steps*steps region
    dy = im.shape[0]  #row
    dx = im.shape[1]
    dst = cv.bilateralFilter(im, 0, 50, 30)  # 双边滤波
    cv.imwrite('../save/'+idx[-6:],dst)
    plt.imshow(dst, cmap='gray')
    plt.show()
    # The feature for each region is the average color of that region.
    # features = []
    # for y in range(dy):
    #     for x in range(dx):
    #         R = im[y , x , 0]/255
    #         G = im[y , x ,1]/255
    #         B = im[y , x , 2]/255
    #         features.append([R, G, B,y/dy,x/dx])
    # features = array(features, 'f')
    #
    # # Cluster.
    # centroids, variance = kmeans(features, 2)
    # code, distance = vq(features, centroids)
    # code = code * 255
    # codeim = code.reshape(dy, dx)
    # codeim = Image.fromarray(codeim).resize(im.shape[:2]).convert('1')
    # codeim.save('../save/'+idx[-6:])
    # figure()
    # imshow(codeim)
    # show()
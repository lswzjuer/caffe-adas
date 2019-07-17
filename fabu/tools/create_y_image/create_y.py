# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-07-09 12:50:53
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-07-09 12:57:44
import cv2
import numpy as np
import os 
import struct

def cretae_y(root):
    assert os.path.exists(root)
    files=os.listdir(root)
    for i, file in enumerate(files):
        file_path=os.path.join(root,file)
        origimg=cv2.imread(file_path)
        if origimg is not None:
            img = origimg.transpose((2,0,1))      
            new_file_path = file_path.replace('jpg','y')
            with open(new_file_path,'wb') as f:
                for x in img.flatten():
                    a = struct.pack('B', x)
                    f.write(a)

if __name__ == '__main__':
    root=r"./y_image"
    cretae_y(root)

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class get_sift:
    def __init__(self,path):
        self.path = path

    def getsift(self):

        final_kp1 = []
        final_kp2 = []
        dp_bowg = []

        list_images = os.listdir(self.path)
        sift = cv2.xfeatures2d.SIFT_create()

        for im_num in range(len(list_images)-1):

            list_kp1 = []
            list_kp2 = []

            img1 = cv2.imread(self.path + list_images[im_num])
            img2 = cv2.imread(self.path + list_images[im_num+1])

            kps1, d1 = sift.detectAndCompute(img1, None)
            kps2, d2 = sift.detectAndCompute(img2, None)

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(d1, d2)

            for mat in matches:
                img1_idx = mat.queryIdx
                img2_idx = mat.trainIdx
                list_kp1.append(kps1[img1_idx].pt)
                list_kp2.append(kps2[img2_idx].pt)

            final_kp1.append(list_kp1)
            final_kp2.append(list_kp2)

            dp_bowg.extend(d1)


        return final_kp1,final_kp2,dp_bowg

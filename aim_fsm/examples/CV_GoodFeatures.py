"""
  CV_GoodFeatures demonstrates the Shi and Tomasi (1994) feature
  extractor built in to OpenCV.
"""

import cv2
import numpy as np
from aim_fsm import *

class CV_GoodFeatures(StateMachineProgram):
    def __init__(self):
        super().__init__(aruco=False,
                         particle_filter=False, 
                         launch_cam_viewer=False,
                         launch_worldmap_viewer=False,
                         force_annotation=True,
                         annotate_sdk=False,
                         speech=False)

    def start(self):
        super().start()
        self.colors = np.random.randint(0,255,(101,3),dtype=np.uint8)
        self.features_image = np.array([[0]],dtype='uint8')

        cv2.namedWindow('features')
        cv2.imshow('features',self.features_image)
        cv2.createTrackbar('maxCorners','features',50,100,lambda self: None)
        cv2.createTrackbar('qualityLevel','features',10,1000,lambda self: None)
        cv2.createTrackbar('minDistance','features',5,50,lambda self: None)

        while True:
            cv2.imshow('features',self.features_image)
            cv2.waitKey(1)

    def user_image(self,image,gray):
        try:
            maxCorners = max(1,cv2.getTrackbarPos('maxCorners','features'))
        except:
            return
        quality = max(1,cv2.getTrackbarPos('qualityLevel','features'))
        cv2.setTrackbarPos('qualityLevel', 'features', quality) # don't allow zero
        minDist = max(1,cv2.getTrackbarPos('minDistance','features'))
        cv2.setTrackbarPos('minDistance', 'features', minDist) # don't allow zero
        qualityLevel = quality / 1000
        corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDist)
        (x,y,_) = image.shape
        for corner in corners:
            x,y = corner.ravel()
            x = int(x); y = int(y)
            color_index = (x+y) % self.colors.shape[0]
            color = self.colors[color_index].tolist()
            cv2.circle(image, (x, y), 4, color, -1)
        self.features_image = image

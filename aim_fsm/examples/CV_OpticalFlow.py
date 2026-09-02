"""
  CV_OpticalFlow demonstrates the Lucas and Kanade optical flow
  algorithm built in to OpenCV.
"""

import cv2
import numpy as np
from aim_fsm import *

class CV_OpticalFlow(StateMachineProgram):
    def __init__(self):
        super().__init__(aruco=False,
                         particle_filter=False, 
                         launch_cam_viewer=False,
                         launch_worldmap_viewer=False,
                         force_annotation=True,
                         annotate_sdk=False,
                         speech=False)

    def start(self):
        self.feature_params = dict( maxCorners = 100,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )

        self.lk_params = dict( winSize = (15,15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS |
                                           cv2.TERM_CRITERIA_COUNT,
                                           10, 0.03) )

        self.colors = np.random.randint(0, 255, (100,3), dtype=np.uint8)

        super().start()
        self.prev_gray = None
        self.good_new = None
        self.mask = None
        self.image = np.array([[0]],dtype='uint8')
        cv2.namedWindow('OpticalFlow')

        while True:
            cv2.imshow('OpticalFlow', self.image)
            cv2.waitKey(1)


    def user_image(self,image,gray):
        try:
            self.prev_gray
        except:
            return
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_feat = cv2.goodFeaturesToTrack(gray, mask=None,
                                                     **self.feature_params)
            return
        new_feat, status, err = \
                  cv2.calcOpticalFlowPyrLK(self.prev_gray, gray,
                                           self.prev_feat, None, **self.lk_params)
        if new_feat is None:
            self.good_new = None
            return
        self.good_new = new_feat[status == 1]
        self.good_old = self.prev_feat[status == 1]
        self.prev_gray = gray
        self.prev_feat = self.good_new.reshape(-1,1,2)

        if self.mask is None:
            self.mask = np.zeros_like(image)

        for i,(new,old) in enumerate(zip(self.good_new, self.good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)
            self.mask = cv2.line(self.mask, (a,b), (c,d),
                                 self.colors[i].tolist(), 2)
            cv2.circle(image,(a,b),5,self.colors[i].tolist(),-1)
        image = cv2.add(image,self.mask)
        self.image = image

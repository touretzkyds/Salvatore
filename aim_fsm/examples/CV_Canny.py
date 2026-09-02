"""
  CV_Canny demonstrates image thresholding in OpenCV, and
  independently, the Canny edge detector.
"""

import cv2
import numpy as np
from aim_fsm import *

class CV_Canny(StateMachineProgram):
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
        dummy = np.array([[0]], dtype='uint8')
        self.thresholded_image = dummy
        self.edges_image = dummy

        cv2.namedWindow('edges')
        cv2.imshow('edges',dummy)

        cv2.namedWindow('threshold')
        cv2.imshow('threshold',dummy)

        cv2.createTrackbar('thresh','threshold',0,255,lambda self: None)
        cv2.setTrackbarPos('thresh', 'threshold', 100)

        cv2.createTrackbar('minval','edges',0,255,lambda self: None)
        cv2.createTrackbar('maxval','edges',0,255,lambda self: None)
        cv2.setTrackbarPos('minval', 'edges', 50)
        cv2.setTrackbarPos('maxval', 'edges', 150)

        while True:
            cv2.imshow('threshold',self.thresholded_image)
            cv2.imshow('edges',self.edges_image)
            cv2.waitKey(1)

    def user_image(self,image,gray):
        # Thresholding
        try:
            self.thresh = cv2.getTrackbarPos('thresh','threshold')
        except:
            return
        ret, self.thresholded_image = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY)

        # Canny edge detection
        try:
            self.minval = cv2.getTrackbarPos('minval','edges')
            self.maxval = cv2.getTrackbarPos('maxval','edges')
        except:
            return
        self.edges_image = cv2.Canny(gray, self.minval, self.maxval, apertureSize=3)

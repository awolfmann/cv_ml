# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
np.seterr(all='raise')

import cv2

from skimage.measure import ransac
from skimage.transform import AffineTransform

if __name__ == "__main__":
    DETECTOR = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, edgeThreshold=1, fastThreshold=10)
    DESCRIPTOR = DETECTOR
    MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    img1 = cv2.imread('ukbench00001.jpg', cv2.IMREAD_GRAYSCALE)
    kp1 = DETECTOR.detect(img1, None)
    kp1, desc1 = DESCRIPTOR.compute(img1, kp1)

    img2 = cv2.imread('ukbench00002.jpg', cv2.IMREAD_GRAYSCALE)
    kp2 = DETECTOR.detect(img2, None)
    kp2, desc2 = DESCRIPTOR.compute(img2, kp2)

    matches = MATCHER.match(desc1, desc2)

    dist_threshold = 64
    good_matches = [m for m in matches if m.distance < dist_threshold]

    kp1_good = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    kp2_good = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    afftr, inliers = ransac((kp1_good, kp2_good), AffineTransform, min_samples=3, residual_threshold=3)

    visu = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
    cv2.imshow("before", visu)
    cv2.waitKey(-1)

    good_matches = [m for i, m in enumerate(good_matches) if inliers[i]]

    visu = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
    cv2.imshow("after", visu)
    cv2.waitKey(-1)

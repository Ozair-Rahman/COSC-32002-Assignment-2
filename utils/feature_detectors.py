import cv2
import numpy as np

def harris_corner_detector(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    # Thresholding to get the corners
    corners = np.argwhere(dst > threshold * dst.max())
    print("Detected corners:", corners.tolist())
    return corners.tolist()  # Return list of corner coordinates

def shi_tomasi_corner_detector(image, max_corners=100, quality_level=0.01, min_distance=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    return corners.tolist()

def fast_detector(image, threshold=10, nonmax_suppression=True):
    fast = cv2.FastFeatureDetector_create(threshold, nonmaxSuppression=nonmax_suppression)
    keypoints = fast.detect(image, None)
    return [kp.pt for kp in keypoints]  # Return list of keypoint coordinates

def orb_detector(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
# Extracting the coordinates of the keypoints
    keypoint_coords = [kp.pt for kp in keypoints]
    
    return keypoint_coords  # Return list of keypoint coordinates and descriptors

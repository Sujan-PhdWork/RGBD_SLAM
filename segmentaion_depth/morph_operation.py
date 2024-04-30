import numpy as np
import cv2


def morph_operation(img: np.ndarray) -> np.ndarray:
     
# img = cv.imread('j.png', cv.IMREAD_GRAYSCALE)
    kernel = np.ones((9,9),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((1,1),np.uint8)
    # closing = cv2.dilate(closing,kernel,iterations = 2)



    return closing
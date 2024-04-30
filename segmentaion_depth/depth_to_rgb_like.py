import numpy as np
import skimage.exposure
import cv2

#for depth mapping to 8bit value
def depth_to_rgb_like(frame: np.ndarray) -> np.ndarray:


    stretch = skimage.exposure.rescale_intensity(frame, in_range='image', out_range=(0,255)).astype(np.uint8)
    stretch = cv2.merge([stretch,stretch,stretch])

    color1 = (0, 0, 255)     #red
    color2 = (0, 165, 255)   #orange
    color3 = (0, 255, 255)   #yellow
    color4 = (255, 255, 0)   #cyan
    color5 = (255, 0, 0)     #blue
    color6 = (128, 64, 64)   #violet
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
    lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)
    result = cv2.LUT(stretch, lut)   
    
    return result
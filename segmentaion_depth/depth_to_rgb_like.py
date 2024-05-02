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
    color7 = (0, 255, 0)     #green
    color8 = (255, 128, 0)   #dark orange
    color9 = (128, 0, 128)   #purple

    colorArr = np.array([
        [color1, color2, color3, color4, color5, color6],  # Shallow depth
        [color7, color8, color3, color4, color9, color6],  # Medium depth
        [color5, color8, color9, color4, color1, color2]   # Deep depth
    ], dtype=np.uint8)

    print(len(colorArr[0]))
    lut = cv2.resize(colorArr, (256, 1), interpolation=cv2.INTER_LINEAR)
    result = cv2.LUT(stretch, lut)




    # color1 = (0, 0, 255)     #red
    # color2 = (0, 165, 255)   #orange
    # color3 = (0, 255, 255)   #yellow
    # color4 = (255, 255, 0)   #cyan
    # color5 = (255, 0, 0)     #blue
    # color6 = (128, 64, 64)   #violet
    # colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
    # lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)
    # result = cv2.LUT(stretch, lut)   
    
    return result
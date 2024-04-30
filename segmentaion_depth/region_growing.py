import numpy as np
import cv2

def region_growing(frame: np.ndarray)->np.ndarray:
    
    r,g,b = cv2.split(frame)

    ddepth = cv2.CV_64F
    
    d_xr = cv2.Sobel(r, ddepth, 1, 0, ksize=3) 
    d_yr = cv2.Sobel(r, ddepth, 0, 1, ksize=3)

    d_xg = cv2.Sobel(g, ddepth, 1, 0, ksize=3) 
    d_yg = cv2.Sobel(g, ddepth, 0, 1, ksize=3)

    d_xb = cv2.Sobel(b, ddepth, 1, 0, ksize=3) 
    d_yb = cv2.Sobel(b, ddepth, 0, 1, ksize=3)


    p=d_xr**2+d_xg**2+d_xb**2
    q=d_yr**2+d_yg**2+d_yb**2

    t=d_xr*d_yr+d_xg*d_yg+d_xb*d_yb

    Lam=0.5*(p+q+np.sqrt((p+q)**2-4*(p*q-t**2)))

    G=np.sqrt(Lam)

    cv2.imshow("colormap_Grad",G)



import numpy as np
import cv2

def region_growing(frame: np.ndarray)->np.ndarray:
    
    grad_map=color_gradient_map(frame)
    
    G_E=gray_level_contrast(grad_map)
    otsu_th= otsu_calc(G_E)

    L_map=Line_filed_map(G_E)

    T_map= Thresh_map(G_E,otsu_th)

    G_W=cv2.bitwise_or(L_map,T_map, mask=None)

    G_W=G_E* G_W

    # Section3.2seed point selection
 

    # otsu_th=12.00
    # grad_copy_map=grad_map.copy()

    # grad_copy_map[grad_map<0.09*otsu_th]=0
    # grad_copy_map[grad_map>0.09*otsu_th]=255   

    
    

    cv2.imshow("colormap_Grad",G_W)


def color_gradient_map(frame):
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
    return G


def otsu_calc(grad_map):

    # grad_map=np.round(grad_map,0)
    # grad_map=grad_map.astype(np.uint16)


    height,width=grad_map.shape

    vec_grad= grad_map.reshape(1,-1)

    #Total number of pixel
    n= height*width

    # distinct gradient value 
    distnict_numbers= np.unique(vec_grad)

    print(distnict_numbers[-1])

    L=distnict_numbers.shape[0]

    # ret, thresh1 = cv2.threshold(grad_map, 0, distnict_numbers[-1], cv2.THRESH_BINARY + 
    #                                         cv2.THRESH_OTSU)
    

    # cv2.imshow("thresold",thresh1)
    # print(thresh1[thresh1!=0])


     # Calculate histogram of the image
    hist, bins = np.histogram(vec_grad, bins=L, range=(0, distnict_numbers[-1]))
    
    # print(hist)
    

    normalized_hist=hist/n

    weight_b = 0  # Background weight
    weight_f = 0  # Foreground weight
    mu_b = 0  # Mean intensity of background
    mu_t = 0  # Mean intensity of total image
    between_class_var_max = 0  # Maximum between-class variance
    threshold = 0  # Optimal threshold

    #   # Calculate total mean intensity
    for i in range(L):
        mu_t += distnict_numbers[i] * normalized_hist[i]

    for i in range(L):
        weight_b += normalized_hist[i]

        mu_b = (1.0/weight_b) * np.sum(distnict_numbers[:i] * normalized_hist[:i])


        weight_f = 1 - weight_b
        if weight_f==0:
            between_class_var=(weight_b/n)*(mu_b-mu_t)**2
        else:    
            # mu_f = (mu_t - mu_b *weight_b) /weight_f

            mu_f = (1.0/weight_f) * np.sum(distnict_numbers[i:] * normalized_hist[i:])

            between_class_var=weight_b*(mu_b-mu_t)**2+weight_f*(mu_f-mu_t)**2

            # between_class_var = weight_b  * weight_f   * (mu_b - mu_f) ** 2

        # print(mu_t, weight_b*mu_b+weight_f*mu_f)

        if between_class_var > between_class_var_max:
            between_class_var_max = between_class_var
            threshold = distnict_numbers[i]

    print(threshold)

    return threshold
        
def gray_level_contrast(grad_map: np.ndarray):

    max_value=np.max(grad_map)
    normalized_image = np.where(grad_map < 0.1 * max_value, 0, grad_map)
    normalized_image = np.where(grad_map > 0.9 * max_value, max_value, normalized_image)
    normalized_image = (normalized_image - 0.1 * max_value) / ((0.9 - 0.1) * max_value)


    #normalized_image - (low_threshold * max_value): 
    #This subtracts the calculated threshold value 
    #from each pixel in the normalized_image array. 
    #This effectively shifts the range of values 
    #that need to be scaled from low_threshold * max_value to 
    #max_value down to 0 to max_value - (low_threshold * max_value).
    
    return normalized_image


def Line_filed_map(grad_map: np.ndarray, TL=0.1,TH=0.8):

    line_field_map = np.zeros_like(grad_map, dtype=np.uint8)

    # Apply high threshold
    line_field_map[grad_map > TH] = 1

    # Apply low threshold and connectivity check
    for i in range(1, grad_map.shape[0] - 1):
        for j in range(1, grad_map.shape[1] - 1):
            
            
            # Check for low threshold and connected neighbors
            if grad_map[i, j] > TL:
                nl=line_field_map[i - 1, j] == 1 or \
                line_field_map[i + 1, j] == 1 or \
                    line_field_map[i, j - 1] == 1 or \
                        line_field_map[i, j + 1] == 1
                if nl:
                    line_field_map[i, j] = 1

    return line_field_map*255

def Thresh_map(grad_map,otsu_th):
    
    Threshold_map = np.zeros_like(grad_map, dtype=np.uint8)
    Threshold_map[grad_map>0.25*otsu_th]=1
    
    return Threshold_map*255


    


    
        
        
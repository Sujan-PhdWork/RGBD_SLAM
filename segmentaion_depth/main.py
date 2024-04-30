import cv2
import numpy as np
from depth_to_rgb_like import depth_to_rgb_like
from nd_6_sigma import nd_6_sigma
from morph_operation import morph_operation
from region_growing import region_growing



def process_img(frame,depth):
    
    print(depth.shape)

    # rejecting outlier in depth image
    depth_mod=nd_6_sigma(depth, sigma=4)
    
    erode_depth=morph_operation(depth_mod)
    colored_depth=depth_to_rgb_like(erode_depth)
    region_growing(colored_depth)




    # img=colored_depth
    # Z = img.reshape((-1,3))
    # Z = np.float32(Z)
    # # define criteria, number of clusters(K) and apply kmeans()
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    # K = 30
    # ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # # Now convert back into uint8, and make original image
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res2 = res.reshape((img.shape))

    # result = cv2.bitwise_or(colored_depth, res2, mask = None)
    # result=cv2.add(colored_depth,res2)



    
    # cv2.imshow('result',result)
    cv2.imshow('depth',colored_depth)
    cv2.imshow('frame',depth)



if __name__ == "__main__":
    dataset_path='dataset/'
    # dataset_path='../dataset/rgbd_dataset_freiburg1_floor/'
    

    # Loop.lc.event.set()


    frame=cv2.imread(dataset_path+'rgb.png') # 8 bit image
    depth=cv2.imread(dataset_path+'depth.png',-1) # 16 bit monochorme image
        
        
        
    process_img(frame,depth)


    if frame is None:
        print("No frame")
    
    cv2.waitKey(0)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
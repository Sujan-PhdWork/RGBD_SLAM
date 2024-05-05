import cv2
import numpy as np
from utils import data

def process_img(img,depth):
    orb=cv2.ORB_create(nfeatures=3000,scaleFactor=2,nlevels=8,patchSize=21,edgeThreshold=21)
    kp, des = orb.detectAndCompute(img, None)
    imgg=cv2.drawKeypoints(img, kp, None)
    cv2.imshow("frame",imgg)




if __name__ == "__main__":
    dataset_path='../dataset/rgbd_dataset_freiburg3_walking_xyz_validation/'
    # dataset_path='../dataset/rgbd_dataset_freiburg3_walking_static/'
    # dataset_path='../dataset/rgbd_dataset_freiburg1_xyz/'
    depth_paths=dataset_path+'depth.txt'
    dlist=data(depth_paths)

    rgb_paths=dataset_path+'rgb.txt'
    ilist=data(rgb_paths)

    # Loop.lc.event.set()

    for i in range(len(dlist)):

        frame=cv2.imread(dataset_path+ilist[i]) # 8 bit image
        depth=cv2.imread(dataset_path+dlist[i],-1) # 16 bit monochorme image
        
        
        
        process_img(frame,depth)


        if frame is None:
            print("End of frame")
            break
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
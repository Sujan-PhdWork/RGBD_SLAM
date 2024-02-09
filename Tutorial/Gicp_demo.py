
import cv2
from ..utils import *


def process_img(img,depth):

    disp(img,"RGB")
    disp(depth,"Depth")


if __name__ == "__main__":
    
    dataset_path='../dataset/rgbd_dataset_freiburg2_xyz/'

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
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
    # optimize_frame(mapp)
    # mapp.display()
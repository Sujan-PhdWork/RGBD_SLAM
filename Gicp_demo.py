
import cv2
from utils import *
import pcl
import numpy as np
import pcl.pcl_visualization


cx=325.5
fx=253.5

cy=518.0
fy=519.0

s=5000.0
visual = pcl.pcl_visualization.CloudViewing()


def to_3D(fx,fy,depth,cx,cy,s,u,v):

    cloud=pcl.PointCloud()
    z=depth/s
    x=(u-cx)*z/fx
    y=(v-cy)*z/fy

    x=np.ravel(x).reshape(-1,1)
    y=np.ravel(y).reshape(-1,1)
    z=np.ravel(z).reshape(-1,1)

    xyz=np.concatenate((x,y,z),axis=1)
    xyz = xyz[~np.all(xyz == 0, axis=1)]

    xyz=xyz.astype(np.float32)
    cloud.from_array(xyz)
    return cloud







def process_img(img,depth):
    H=depth.shape[0]
    W=depth.shape[1]

    u=np.arange(W)
    v=np.arange(H)

    u, v = np.meshgrid(u, v)

    cloud = to_3D(fx, fy, depth, cx, cy, s, u, v)
    # xyz=xyz[xyz != 0]

    # xyz = xyz[~np.all(xyz == 0, axis=1)]


    
    visual.ShowMonochromeCloud(cloud)



            



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
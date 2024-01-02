import cv2
import numpy as np
from utils import *
from frame import *
import matplotlib.pyplot as plt

import g2o
from pointmap import Point,Map


np.set_printoptions(suppress=True)
# Process_flag=True

mapp=Map()
mapp.create_viewer()


def process_img(img,depth):

    frame=Frame(mapp,img,depth)
    # frames.append(frame)

    if frame.id==0:
        return
    
    f1=mapp.frames[-1]
    f2=mapp.frames[-2]
    idx1,idx2,pose=match(f1,f2)

    
    if idx1 is None:
        return
    if idx2 is None:
        return
    
    if pose is None:
        return
    
    f1.pose=np.dot(pose,f2.pose)

    # print(frames[-1].pts)
    pt4ds=add_ones(f1.pts[idx1])
    # print(pt4d.shape)
    pt4ds=np.dot(f1.pose,pt4ds.T).T[:3]
    # print(idx1)
    # print(pt4ds.shape)

    for i,p in enumerate(pt4ds):
        pt=Point(mapp,p)
        pt.add_observation(f1,idx1[i])
        pt.add_observation(f2,idx2[i])
    
    # print(pt4d[:,0])
    
    

    # print(frames[-1].pose)

    for kp1,kp2 in zip(f1.pts[idx1],f2.pts[idx2]):
        u1,v1,_=map(lambda x: int(round(x)),kp1)
        u2,v2,_=map(lambda x: int(round(x)),kp2)

        cv2.circle(img,(u1,v1),color=(0,255,0),radius=3)
        cv2.line(img,(u1,v1),(u2,v2),color=(255,0,0))
    disp(img,"RGB")



    mapp.display()


if __name__ == "__main__":
    
    dataset_path='../rgbd_dataset_freiburg2_rpy/'

    depth_paths=dataset_path+'depth.txt'
    dlist=data(depth_paths)

    rgb_paths=dataset_path+'rgb.txt'
    ilist=data(rgb_paths)

    

    for i in range(len(dlist)):

        frame=cv2.imread(dataset_path+ilist[i])
        depth=cv2.imread(dataset_path+dlist[i],0)
        # print(frame.shape,depth.shape)


        process_img(frame,depth)
        # print(frame)

        if frame is None:
            print("End of frame")
            break

        
        disp(depth,"Depth")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


        

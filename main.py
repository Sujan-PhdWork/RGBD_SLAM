
import cv2
import numpy as np
from utils import *
from frame import match,Frame,Keyframes
# import g2o
from pointmap import Map,EDGE
from GICP import GICP
from loop_closure import loop_closure,lc_process
from threading import Thread,Lock
from keyframe import Keyframes



np.set_printoptions(suppress=True)

W,H=640,480

# camera parameters
fx=517.3
fy=516.5
cx=318.6
cy=255.3

K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])


# #freiburg1_xyz

# Int_pose=np.array([[0.4630,0.0940,-0.8814,1.3563],
#                    [-0.8837,-0.0287,-0.4672,0.6305],
#                    [-0.0692,0.9952,0.0698,1.6380],
#                    [0,0,0,1]])


#freiburg1_floor
Int_pose=np.array([[0.6053,    0.5335,   -0.5908,    1.2764],
                    [-0.7960,    0.4055,   -0.4493,   -0.9763],
                    [-0.0001,    0.7423,    0.6701,    0.6837],
                     [      0,         0,         0,    1.0000]])


# Int_pose=np.eye(4)

mapp=Map()
mapp.create_viewer()

#keyframe Thresolding
th=300.0

# Computing keyframe
kf=Keyframes()
kf.create_Thread(mapp,th)


def process_img(img,depth):
    
    
    # creating frame object
    frame=Frame(mapp,img,depth,K)
 

    if (frame.id)==0:
        # Adding first frameas key frame 
        mapp.keyframe=frame
        mapp.keyframes.append(frame)
        return
    
    f_c=mapp.frames[-1] #current frame
    f_p=mapp.frames[-2] # previous frame
    
    
    # finding the between consecutive frame 
    idx2,idx1,pose=match(f_c,f_p)
    
    #0 idx1-> id of the keypoint in previous frame
    # idx2-> id of the keypoint in current frame
    
    # pose-> relative transformation of current frame  
    # with respect to previous frame 

    assert len(idx1)>0
    assert len(idx2)>0
    
    if pose is None:
        return
    
    
    #creating a edge between consecutive frame
    f_c.pose=np.dot(pose,f_p.pose) 
    EDGE(mapp,f_p.id,f_c.id,pose)
    
    
    
    print("current keyframe",mapp.keyframes[-1].id)
    
    

    # displaying features on RGB image
    for kp1,kp2 in zip(f_p.kps[idx1],f_c.kps[idx2]):
        u_p,v_p,_=denormalize(kp1,f_p.K)
        u_c,v_c,_=denormalize(kp2,f_c.K)
        # u_p,v_p,_=kp1

        cv2.circle(img,(u_p,v_p),color=(0,255,0),radius=3)
        cv2.line(img,(u_p,v_p),(u_c,v_c),color=(255,0,0))

    
    
    disp(img,"RGB")
    disp(depth,"Depth")


def optimize_frame(mapp):
    mapp.optimize()
    


if __name__ == "__main__":
    
    dataset_path='../dataset/rgbd_dataset_freiburg1_floor/'

    depth_paths=dataset_path+'depth.txt'
    dlist=data(depth_paths)

    rgb_paths=dataset_path+'rgb.txt'
    ilist=data(rgb_paths)



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
    
    optimize_frame(mapp)
    mapp.display()


        




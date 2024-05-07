
import cv2
import numpy as np
from utils import *
from frame import match,Frame,match_by_segmentation
from keyframe import Keyframe
from Local_Mapping import LocalMap_Thread
# import g2o
from pointmap import Map,EDGE
from GICP_test import GICP,GICP_Thread
from loop_closure import Loop_Thread
from threading import Thread,Lock
import pcl.pcl_visualization
import pcl
from Full_map import FulllMap_Thread

# from keyframe import Keyframes
# from  local_mapping import local_mapping  



# viewer = pcl.pcl_visualization.PCLVisualizering()
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
# th1=900.0

# # Computing keyframe
Local_map=LocalMap_Thread()
Local_map.create_Thread(mapp)
kFrame=None

Full_MAP=FulllMap_Thread()
Full_MAP.create_Thread(mapp)
# kFrame=None

th2=0.6
Loop=Loop_Thread()
Loop.create_Thread(mapp,th2)



# GICP_T=GICP_Thread()
# GICP_T.create_Thread(mapp)






def process_img(img,depth):
    global kFrame
    
        
    # creating frame object
    frame=Frame(mapp,img,depth,K)
    
    
    mapp.frames.append(frame)
 
    # print(1)
    if (frame.id)==0:
        # print(2)
        # Adding first frameas key frame 
        # mapp.keyframe=
        # GICP_T.gc.event.set()
        Loop.lc.event.set()
        print("Loop closing initialized")
        frame.pose=Int_pose
        # frame.Rpose=Int_pose
        frame.isKey=True
        kFrame=Keyframe(frame)
        return
    

   
    
    # print(kFrame.pose)
    
    f_c=mapp.frames[-1] #Current frame
    f_p=mapp.frames[-2] # Previous frame


    

    #Matching between current_frame and keyframe
    Kidx2,Kidx1,Kpose=match_by_segmentation(f_c,kFrame.frame)
    
    #initialize pose of each frame
    idx2,idx1,pose=match_by_segmentation(f_c,f_p)

    # f_c.pose=np.dot(Kpose,kFrame.frame.pose)
    f_c.pose=np.dot(pose,f_p.pose)

    # if mapp.keyframes:
    #     print("Keyframe ID: ",len(mapp.keyframes))

    # Take the next frame as an reference for ratio

    if (frame.id-kFrame.id)==1:
        # f_c.pose=Kpose
        kFrame.add_frames(f_c)
        # print(kFrame.id)
        kFrame.nmpts=len(Kidx1)
    
    else:
        with Lock():
            flag=Local_map.lm.Acceptance_flag
            # print(mapp.frames[2].pose)
        # _,_,pose=match(f_c,f_p)
        
        
        
        
        M_ratio=len(Kidx1)/kFrame.nmpts # Matching ratio
        
        if (M_ratio<0.8) and (flag):
            # Local_map.lm.CheckNewKeyframe=True
            with Lock():
                print("Key id:",f_c.id)
                
                Local_map.lm.NewKeyframes.append(kFrame)
                Local_map.lm.SetAcceptKeyFrames(False)
                # Local_map.lm.join()
                # Local_map.event.set()
                # Full_MAP.event.set()
            
            f_c.pose=Kpose
            f_c.isKey=True
            EDGE(mapp,kFrame.id,f_c.id,Kpose,0.02)
            kFrame=Keyframe(f_c)
            
        else:
            kFrame.add_frames(f_c)
        



    # # displaying features on RGB image
    for kp1,kp2 in zip(f_p.kps[idx1],f_c.kps[idx2]):
        u_p,v_p,_=denormalize(kp1,f_p.K)
        u_c,v_c,_=denormalize(kp2,f_c.K)
        # u_p,v_p,_=kp1

        cv2.circle(img,(u_p,v_p),color=(0,255,0),radius=3)
        cv2.line(img,(u_p,v_p),(u_c,v_c),color=(255,0,0))

    
    
    disp(img,"RGB")
    disp(depth,"Depth")
    # frame.id>20:
    mapp.display()
    # del frame
    # mapp.optimize()


def optimize_frame(mapp):
    mapp.optimize()
    


if __name__ == "__main__":
    
    dataset_path='../dataset/rgbd_dataset_freiburg3_walking_xyz_validation/'
    # dataset_path='../dataset/rgbd_dataset_freiburg3_walking_static/'
    # dataset_path='../dataset/rgbd_dataset_freiburg1_floor/'

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


        



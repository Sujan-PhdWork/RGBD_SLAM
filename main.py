
import cv2
import numpy as np
from utils import *
from frame import match,Frame
# import g2o
from pointmap import Map,EDGE
from GICP import GICP
from loop_closure import loop_closure,lc_process
from threading import Thread,Lock



np.set_printoptions(suppress=True)

W,H=640,480
K=np.array([[1,0,W//2],[0,1,H//2],[0,0,1]])

# #freiburg1_xyz

Int_pose=np.array([[0.4630,0.0940,-0.8814,1.3563],
                   [-0.8837,-0.0287,-0.4672,0.6305],
                   [-0.0692,0.9952,0.0698,1.6380],
                   [0,0,0,1]])


#freiburg1_floor
Int_pose=np.array([[0.6053,    0.5335,   -0.5908,    1.2764],
                    [-0.7960,    0.4055,   -0.4493,   -0.9763],
                    [-0.0001,    0.7423,    0.6701,    0.6837],
                     [      0,         0,         0,    1.0000]])


# Int_pose=np.eye(4)

mapp=Map()
mapp.create_viewer()

def process_img(img,depth):
    
    frame=Frame(mapp,img,depth,K)


    if (frame.id)==0:
        frame.pose=Int_pose
        frame.Rpose=Int_pose
        return
    
    f_c=mapp.frames[-1] #current frame
    f_p=mapp.frames[-2] # previous frame
    
    
    idx1,idx2,pose=match(f_p,f_c)
    
    
    # idx1-> id of the keypoint in previous frame
    # idx2-> id of the keypoint in current frame
    # pose relative transformation of current frame  
    # with respect to previous frame 

    assert len(idx1)>0
    assert len(idx2)>0
    
    if pose is None:
        return
    
    f_c.pose=np.dot(pose,f_p.pose) 
    
    # relative orientaion from pose of second frame 
    #with respect to first frame

    f_p.pts=f_p.kps[idx1] # points on previous frame
    f_c.pts=f_c.kps[idx2] # points on current frame

    # print(np.sum(f_c.hist))
    if frame.id >20:
        lc_process(mapp,20)
#     
    # if frame.id>1: 
        
    #     T_pose=GICP(mapp,f_p.id,f_c.id)
    #     # f_c.Rpose=np.dot(T_pose,f_p.Rpose)     
    #     EDGE(mapp,f_p.id,f_c.id,T_pose)

    





    # relative orientaion from point of second frame 
    #with respect to first frame


    for kp1,kp2 in zip(f_p.kps[idx1],f_c.kps[idx2]):
        u_p,v_p,_=denormalize(kp1,f_p.K)
        u_c,v_c,_=denormalize(kp2,f_c.K)
        # u_p,v_p,_=kp1

        cv2.circle(img,(u_p,v_p),color=(0,255,0),radius=3)
        cv2.line(img,(u_p,v_p),(u_c,v_c),color=(255,0,0))

    
    
    disp(img,"RGB")
    disp(depth,"Depth")


    #
    # if frame.id % 20 ==1:
    #     mapp.optimize()
    
    # mapp.display()
    



def optimize_frame(mapp):
    mapp.optimize()
    mapp.display()


if __name__ == "__main__":
    
    dataset_path='../dataset/rgbd_dataset_freiburg1_floor/'

    depth_paths=dataset_path+'depth.txt'
    dlist=data(depth_paths)

    rgb_paths=dataset_path+'rgb.txt'
    ilist=data(rgb_paths)



    for i in range(len(dlist)):

        frame=cv2.imread(dataset_path+ilist[i])
        depth=cv2.imread(dataset_path+dlist[i],0)
        # print(frame.shape,depth.shape)


        process_img(frame,depth)

        # t1=Thread(target=process_img,args=(frame,depth))
        # t1.start()
        
        # print(frame)
        
        # if mapp.q is None :
        #     break

        if frame is None:
            print("End of frame")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
    optimize_frame(mapp)


        




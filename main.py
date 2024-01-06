
import cv2
import numpy as np
from utils import *
from frame import extract,match,Frame
# import g2o
from pointmap import Point,Map
from gicp import gicp


np.set_printoptions(suppress=True)

W,H=640,480
K=np.array([[1,0,W//2],[0,1,H//2],[0,0,1]])

mapp=Map()



def process_img(img,depth):

    frame=Frame(mapp,img,depth,K)


    if (frame.id)==0:
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
    
    
    # print(f_c.pose)



    pts3d1=f_p.kps[idx1] # points on previous frame
    pts3d2=f_c.kps[idx2] # points on current frame

    f_p.pts=pts3d1
    f_c.pts=pts3d2
    
    T_pose=gicp(mapp)

    # relative orientaion from point of second frame 
    #with respect to first frame

    
    print("pose")
    print(pose)

    print("point_pose")
    print(T_pose)




    for kp1,kp2 in zip(f_p.kps[idx1],f_c.kps[idx2]):
        u_p,v_p,_=denormalize(kp1,f_p.K)
        u_c,v_c,_=denormalize(kp2,f_c.K)
        # u_p,v_p,_=kp1

        cv2.circle(img,(u_p,v_p),color=(0,255,0),radius=3)
        cv2.line(img,(u_p,v_p),(u_c,v_c),color=(255,0,0))
    disp(img,"RGB")
    # mapp.display()


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
        # print(frame)
        
        # if mapp.q is None :
        #     break

        if frame is None:
            print("End of frame")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


        




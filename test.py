import cv2
import numpy as np
from utils import *
from frame import match,Frame,match_by_segmentation_mod
from pointmap import Map,EDGE



# viewer = pcl.pcl_visualization.PCLVisualizering()
np.set_printoptions(suppress=True)

W,H=640,480

# camera parameters
fx=517.3
fy=516.5
cx=318.6
cy=255.3

K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])


Int_pose=np.eye(4)

mapp=Map()
def process_img(img,depth):
   
    
    frame=Frame(mapp,img,depth,K)
   
            

    mapp.frames.append(frame)
    
    
    if (frame.id)==0:
        frame.pose=Int_pose
        return
    
    f_c=mapp.frames[-1] #Current frame
    f_p=mapp.frames[-2] # Previous frame
    
    
    
    
    idx1,idx2,pose,p_c=match_by_segmentation_mod(f_c,f_p)
    
    # f_c.pose=np.dot(pose,f_p.pose)

    

    for kp1,kp2 in zip(f_p.kps[idx2],f_c.kps[idx1]):
        u_p,v_p,_=denormalize(kp1,f_p.K)
        u_c,v_c,_=denormalize(kp2,f_c.K)
        # u_p,v_p,_=kp1
        r=5
        pt1=(u_c-r,v_c-r)
        pt2=(u_c+r,v_c+r)
                

        cv2.rectangle(img,pt1,pt2,(0,0,255))
       
        cv2.circle(img,(u_c,v_c),color=(0,255,0),radius=3)
        cv2.line(img,(u_c,v_c),(u_p,v_p),color=(255,0,0))

    for i in range(p_c.shape[0]):
       u_c,v_c= denormalize2(p_c[i],f_p.K)
       cv2.circle(img,(u_c,v_c),color=(0,255,255),radius=3,thickness=-1)



    

    disp(img,"RGB")
    disp(depth,"Depth")
    disp(f_c.colored_segmented_img,"kmean")



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
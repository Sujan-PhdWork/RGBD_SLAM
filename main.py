
import cv2
import numpy as np
from utils import *
from frame import extract,match,Frame
# import g2o
from pointmap import Point,Map


np.set_printoptions(suppress=True)

W,H=640,480
K=np.array([[1,0,W//2],[0,1,H//2],[0,0,1]])
# K=np.array([[1,0,0],[0,1,0],[0,0,1]])
th=0


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


mapp=Map()
mapp.create_viewer()



def process_img(img,depth):

    frame=Frame(mapp,img,depth,K)


    if (frame.id)==0:
        frame.pose=Int_pose
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
    


    for i,idx in enumerate(idx1):
        if f_p.pts[idx] is not None:
            f_p.pts[idx].add_observation(f_c,idx2[i])

    f_c.pose=np.dot(pose,f_p.pose)
    # print(f_c.pose)

    pts4d=add_ones(f_c.kps[idx2])
    pts4d=np.dot(f_c.pose,pts4d.T).T
    

    unmatched_points=np.array([f_c.pts[i] is None for i in idx2])
    #This ensure the point dont has any corospondenc
    
    good_pts4d = (pts4d[:, 2] > 0) & unmatched_points
    
    for i, pt in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt=Point(mapp,pt)
        pt.add_observation(f_p,idx1[i])
        pt.add_observation(f_c,idx2[i])
    
    # print(mapp.points[0].pt)


    for kp1,kp2 in zip(f_p.kps[idx1],f_c.kps[idx2]):
        u_p,v_p,_=denormalize(kp1,f_p.K)
        u_c,v_c,_=denormalize(kp2,f_c.K)
        # u_p,v_p,_=kp1

        cv2.circle(img,(u_p,v_p),color=(0,255,0),radius=3)
        cv2.line(img,(u_p,v_p),(u_c,v_c),color=(255,0,0))
    disp(img,"RGB")

    if frame.id >= 4:
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
        # print(frame)
        
        if mapp.q is None :
            break

        if frame is None:
            print("End of frame")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


        




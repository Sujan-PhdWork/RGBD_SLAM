import cv2
import numpy as np
from transformation import *
from ransac import *
from utils import normalize
import joblib
import pcl
from segmentation import segmentation
from PoseOptimization import PoseOptimization

# model_filename = 'BoW/kmeans_model.joblib'
# kmeans_loaded = joblib.load(model_filename)


IRt=np.eye(4)

def extractRt(model):
    R=model.params['R']
    t=model.params['t']
    c=model.params['c']

    pose=np.eye(4)
    u,s,vh = np.linalg.svd(c*R)
    pose[:3,:3]=u @ vh
    pose[:3,3]=t.reshape(3)

    return pose

def to_3D(depth,K):
  
    fx=K[0,0]
    cx=K[0,2]

    fy=K[1,1]
    cy=K[1,2]


    H=depth.shape[0]
    W=depth.shape[1]

    u=np.arange(W)
    v=np.arange(H)

    u, v = np.meshgrid(u, v)

    
    z=depth/5000.0
    x=(u-cx)*z/fx
    y=(v-cy)*z/fy

    x=np.ravel(x).reshape(-1,1)
    y=np.ravel(y).reshape(-1,1)
    z=np.ravel(z).reshape(-1,1)

    xyz=np.concatenate((x,y,z),axis=1)
    xyz = xyz[~np.all(xyz == 0, axis=1)]

    xyz=xyz.astype(np.float32)
    return xyz


def extract(img,depth,label_img):
    
    orb=cv2.ORB_create(nfeatures=10000,scaleFactor=1.5,nlevels=8,patchSize=21,edgeThreshold=21)
    kps= orb.detect(img, None)
    
    # feats=cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
    # feats=cv2.goodFeaturesToTrack(img.astype(np.uint8),7000,qualityLevel=0.01,minDistance=3)

    # print(feats)
    # kps=[cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in feats]
    
    # _size in numpy<1.75
    
    kps,des=orb.compute(img,kps)

    
    
    # print(des.shape)
    modified_kps=[]
    modified_des=[]
    modified_label=[]        
    
    for i,kp in enumerate(kps):
        u,v=map(lambda x: int(x),kp.pt)
        z=depth[v,u]
        l=label_img[v,u]
        if z!=0:
            modified_kps.append(kp)
            modified_des.append(des[i])
            modified_label.append(l)   
    
    modified_des=np.array(modified_des)


    

    return np.array([(kp.pt[0],kp.pt[1],
                      depth[int(round(kp.pt[1])),int(round(kp.pt[0]))]) for kp in modified_kps]),modified_des,np.array([modified_label])


def triangulate(pose1,pose2,pts1,pts2):
     ret=np.zeros((pts1.shape[0],4))
     pose1=np.linalg.inv(pose1)
    #  pose2=np.linalg.inv(pose2)
     for i, p in enumerate(zip(pts1,pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _,_,vt=np.linalg.svd(A)
        ret[i]=vt[3]
         
     return ret.T


def frames_triangulation(f1, f2, idx1, idx2, pose):
    
    
    kps1 = f1.kps[idx1].copy()
    kps2 = f2.kps[idx2].copy()

    # print(kps1,kps2)
    
    kps1 /= kps1[:, 2].reshape(-1, 1)+1e-06
    kps2 /= kps2[:, 2].reshape(-1, 1)+1e-06
    
   

    kps1 = kps1[:, :2]
    kps2 = kps2[:, :2]

    
    
    P1=np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    # Triangulate the points

    pts4d=triangulate(np.eye(4),pose,kps2,kps1)
    pts4d /= pts4d[3, :]

    # # Project the triangulated points back to the image plane
    projected_pts1 = np.dot(P1, pts4d)
    projected_pts2 = np.dot(pose[:3, :4], pts4d)

    # # Normalize the projected points
    projected_pts1 /= projected_pts1[2, :]
    projected_pts2 /= projected_pts2[2, :]
    
    return pts4d,projected_pts2.T 



def match_by_segmentation(f1,f2):
    
    
    # f1 is current frame f2 is previous frame
    ## we will first check how many labels are in last frame
    ## we will use all the points to calculate the transformation
    # we will use this transformation to calculate the projection matrix of the each frames
    
    # we will then check the matched point triangulation depth and original depth
    # if the covariance is less than threshold we will consider it as a good match
    # then we will take all the labels except an random one
    # and check if covariance is increase or decrease if its increase then we will reject all the point that are in that label
    # if its decreases we will accept all the points that are in that label and make that segment as good segment


    
    n_label=np.unique(f1.label)
    
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.knnMatch(f1.des,f2.des,k=2)
    ret=[]
    idx1,idx2=[],[]

    pose=None
    for m,n in matches:
        if m.distance <0.75*n.distance:
            if m.distance < 30:
                

                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                
                kp1=f1.kps[m.queryIdx]
                kp2=f2.kps[m.trainIdx]

                # kp1[idx]-> the keypoint in previous frame with id =idx            
                ret.append((kp2,kp1))
    
    ret=np.array(ret).astype(np.float32)
    idx1=np.array(idx1)
    idx2=np.array(idx2)
    
    ransac=RANSAC(ret,Transformation(),8,0.01,100)
    model,inliers,error=ransac.solve()
    ret=ret[inliers]
    idx1=idx1[inliers]
    idx2=idx2[inliers]

    idx1_t=idx1.copy()
    idx2_t=idx2.copy()

    pose=extractRt(model)
    # pose=np.linalg.inv(pose)
    point_p,_=frames_triangulation(f1, f2, idx1, idx2, pose)

    error_list=[]

    diff=ret[:,1,:]-point_p[:3,:].T
    
    # errors=np.sqrt(np.sum(diff*diff, axis=1))
    # mean = np.mean(errors)
    # std = np.std(errors)

    # # Calculate lower and upper bounds for 3 sigma
    # lower_bound = mean - 6 * std
    # upper_bound = mean + 6 * std

    # mask=(errors >= lower_bound) & (errors <= upper_bound)

    # idx1=idx1[mask]  
    # idx2=idx2[mask]
    # # pt_proj_c=pt_proj_c[mask]
    # # print(errors[~mask])
    # diff=diff[mask]
    initial_error=np.sqrt(np.mean(np.sum(diff*diff, axis=1),axis=0))
    
    

    error_list.append(initial_error)
    # print(initial_error)

    if initial_error<2.0:
        return idx1,idx2,pose


    else:
        # masks=f1.label==0
        # masks=masks.reshape(-1,1)
        # ret=ret[masks[:,0]]
        # idx1=idx1[masks[:,0]]
        # idx2=idx2[masks[:,0]]
        # ransac=RANSAC(new_ret,Transformation(),8,0.01,100)
        # model,inliers,error=ransac.solve()
        # ret=ret[inliers]
        # idx1=idx1[inliers]
        # idx2=idx2[in=
        for i in range(len(n_label)):
            if i==0:
                continue
            masks=f1.label!=i
            masks=masks.reshape(-1,1)
            
            masks=masks[idx1]
            new_ret=ret[masks[:,0]]
            # print(new_ret)
            new_idx1=idx1[masks[:,0]]
            new_idx2=idx2[masks[:,0]]
            ransac=RANSAC(new_ret,Transformation(),8,0.01,100)
            model,inliers,error=ransac.solve()
                
            new_ret=new_ret[inliers]
            new_idx1=new_idx1[inliers]
            new_idx2=new_idx2[inliers]


            pose=extractRt(model)
            # pose=np.linalg.inv(pose)
            point_p,_=frames_triangulation(f1, f2, new_idx1, new_idx2, pose)

            diff=new_ret[:,1,:]-point_p[:3,:].T
            # errors=np.sqrt(np.sum(diff*diff, axis=1))
            # mean = np.mean(errors)
            # std = np.std(errors)

            # # Calculate lower and upper bounds for 3 sigma
            # lower_bound = mean - 6 * std
            # upper_bound = mean + 6 * std

            # mask=(errors >= lower_bound) & (errors <= upper_bound)

            # idx1=idx1[mask]  
            # idx2=idx2[mask]
            # # pt_proj_c=pt_proj_c[mask]
            # # print(errors[~mask])
            # diff=diff[mask]
            error=np.sqrt(np.mean(np.sum(diff*diff, axis=1),axis=0))
            error_list.append(error)
            
            
            if error > initial_error:
                print("Good lebel",i)
                continue
            else:
                if error<0.2:
                    initial_error=error
                print("deleting label: ", i)
                initial_error=error
                ret=new_ret.copy()
                idx1=new_idx1.copy()
                idx2=new_idx2.copy()
                # initial_error=error
            # else:
            #     print("deleting label outer: ", i)
            #     ret=new_ret
            #     idx1=new_idx1
            #     idx2=new_idx2

                # print("deleting lebel ",i, eg[0])
                
                
                # if eg[0]<0.5:
                #     return idx1,idx2,pose
            
            

    # print(error_list)
    return idx1,idx2,pose
        # f1_label=f1.








def match_by_segmentation_mod(f1,f2):
    
    
    # f1 is current frame f2 is previous frame
    ## we will first check how many labels are in last frame
    ## we will use all the points to calculate the transformation
    # we will use this transformation to calculate the projection matrix of the each frames
    
    # we will then check the matched point triangulation depth and original depth
    # if the covariance is less than threshold we will consider it as a good match
    # then we will take all the labels except an random one
    # and check if covariance is increase or decrease if its increase then we will reject all the point that are in that label
    # if its decreases we will accept all the points that are in that label and make that segment as good segment


    
    n_label=np.unique(f1.label)
    
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.knnMatch(f1.des,f2.des,k=2)
    ret=[]
    idx1,idx2=[],[]

    pose=None
    for m,n in matches:
        if m.distance <0.75*n.distance:
            if m.distance < 30:
                

                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                
                kp1=f1.kps[m.queryIdx]
                kp2=f2.kps[m.trainIdx]

                # kp1[idx]-> the keypoint in previous frame with id =idx            
                ret.append((kp2,kp1))
    
    ret=np.array(ret).astype(np.float32)
    idx1=np.array(idx1)
    idx2=np.array(idx2)
    
    ransac=RANSAC(ret,Transformation(),8,0.05,100)
    model,inliers,error=ransac.solve()
    ret=ret[inliers]
    idx1=idx1[inliers]
    idx2=idx2[inliers]

    Pose=extractRt(model)
    # Pose=np.linalg.inv(Pose)

    point_p,pt_proj_c=frames_triangulation(f1, f2.kps, idx1, idx2, Pose)
    diff=ret[:,1,:]-point_p[:3,:].T

    errors=np.sqrt(np.sum(diff*diff, axis=1))
    
    mean = np.mean(errors)

    std = np.std(errors)

    # Calculate lower and upper bounds for 3 sigma
    lower_bound = mean - 6 * std
    upper_bound = mean + 6 * std

    mask=(errors >= lower_bound) & (errors <= upper_bound)

    idx1=idx1[mask]  
    idx2=idx2[mask]
    pt_proj_c=pt_proj_c[mask]
    # print(errors[~mask])

    





    diff=diff[mask]
    initial_error=np.sqrt(np.mean(np.sum(diff*diff, axis=1),axis=0))
    print(initial_error)
    
    # Pose2=PoseOptimization(f1,f2,idx1,idx2,Pose)
    
    # print(np.dot(Pose,np.linalg.inv(Pose2)))

    # print(Pose[:3,3])
    



    
    
    
    
    
    
    
    # good_pts4d=point_p[3,:]>0 & (np.abs(point_p[2,:])>0.005)


    # point_p=point_p[:,good_pts4d]
    # ret=ret[good_pts4d]
    # idx1=idx1[good_pts4d]
    # idx2=idx2[good_pts4d]
    # pt_proj_c=pt_proj_c[good_pts4d]
    
    # mask=f2.label!=0
    # mask=mask.reshape(-1,1)
            
    # mask=mask[idx2]
    # ret=ret[~mask[:,0]]
    # idx1=idx1[~mask[:,0]]
    # idx2=idx2[~mask[:,0]]

    # pt_proj_c=pt_proj_c[~mask[:,0]]



    return idx1,idx2,Pose,pt_proj_c
        # f1_label=f1.







def match(f1,f2):
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.knnMatch(f1.des,f2.des,k=2)
    
    #low's ratio test
    ret=[]
    idx1,idx2=[],[]
    pose=None

    for m,n in matches:
        if m.distance <0.75*n.distance:
            if m.distance < 30:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

                # idx1-> id of the keypoint in previous frame
                # idx2-> id of the keypoint in current frame

                kp1=f1.kps[m.queryIdx]
                kp2=f2.kps[m.trainIdx]

                # kp1[idx]-> the keypoint in previous frame with id =idx            
                ret.append((kp2,kp1))

    assert len(ret)>=3
    ret=np.array(ret).astype(np.float32)


    idx1=np.array(idx1)
    idx2=np.array(idx2)

    # ret[:,0,:2]=normalize(ret[:,0,:2],f1.Kinv)
    # ret[:,1,:2]=normalize(ret[:,1,:2],f2.Kinv)
    
    ransac=RANSAC(ret,Transformation(),8,0.01,100)
    model,inliers,error=ransac.solve()
    # print(error)
    idx1=idx1[inliers]
    idx2=idx2[inliers]

    pose=extractRt(model)

    # pose=np.linalg.inv(pose)
    return idx1,idx2,pose


def GICP(cloud2,cloud1):
    icp = cloud1.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(cloud1, cloud2)
    # print('has converged:' + str(converged) + ' score: ' + str(fitness))
    # print(str(transf))
    return transf
    






class Frame(object):
    def __init__(self,mapp,img,depth,K):
        
        
        self.cloud=to_3D(depth,K)

        

        label_img,self.colored_segmented_img,self.background=segmentation(img,viz=True)
        
        self.pts,self.des,self.label=extract(img,depth,label_img)
        # print(self.des.shape,self.label.shape)
        self.mod_des=self.des[self.label.flatten()==0]



        # center = np.uint8(center)
        # print(label.shape, depth_Z.shape,depth.shape)
        # res = center[label.flatten()]
        # res2 = res.reshape((depth.shape))
        # cv2.imshow('res2',res2) 


        self.Ihist=None
        #pts is 3d points unlormalize point

        self.K=K

        
        self.Kinv=np.linalg.inv(K)

        self.kps=self.pts.copy()
        # self.pts[:,:,2]=normalize(self.kps[:,:2],self.Kinv)

        self.kps[:,:2]=normalize(self.kps[:,:2],self.Kinv)
        
        #kps is 3d  normalize point 
        self.kps[:,2]=self.kps[:,2]/5000.0 #factor 
        
        self.kps[:,0]=self.kps[:,0]*self.kps[:,2]
        self.kps[:,1]=self.kps[:,1]*self.kps[:,2]

        self.pose=IRt
        self.Rpose=IRt
        self.isKey=False
        self.id=len(mapp.frames)
        







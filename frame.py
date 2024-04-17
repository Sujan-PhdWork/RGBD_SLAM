import cv2
import numpy as np
from transformation import *
from ransac import *
from utils import normalize
import joblib
import pcl

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
    
    orb=cv2.ORB_create()
    feats=cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
    # feats=cv2.goodFeaturesToTrack(img.astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)

    # print(feats)
    kps=[cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in feats]
    
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


def sanity_check_triangulation(f1, f2, idx1, idx2, pose):
    # Extract the keypoints and descriptors for the corresponding indices
    # print(f1.kps[idx1].shape)
    
    kps1 = f1.kps[idx1]
    kps2 = f2.kps[idx2]

    kps1 /= kps1[:, 2].reshape(-1, 1)
    kps2 /= kps2[:, 2].reshape(-1, 1)

    kps1 = kps1[:, :2]
    kps2 = kps2[:, :2]

    
    # des1 = f1.des[idx1]
    # des2 = f2.des[idx2]
    P1=np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    # Triangulate the points

    pts4d=triangulate(np.eye(4),pose,kps2,kps1)
    # pts4d = cv2.triangulatePoints(P1, pose[:3, :4], kps2.T, kps1.T)
    pts4d /= pts4d[3, :]

    # Project the triangulated points back to the image plane
    projected_pts1 = np.dot(P1, pts4d)
    projected_pts2 = np.dot(pose[:3, :4], pts4d)

    # Normalize the projected points
    projected_pts1 /= projected_pts1[2, :]
    projected_pts2 /= projected_pts2[2, :]

    # print("a")
    # print(kps1)
    # print("Hi")
    # print(projected_pts1.T)
    # print("b")
    return projected_pts1.T, projected_pts2.T



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


    
    frame_lable=np.unique(f2.label)
    
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
                ret.append((kp1,kp2))
    
    ret=np.array(ret).astype(np.float32)
    idx1=np.array(idx1)
    idx2=np.array(idx2)
    
    ransac=RANSAC(ret,Transformation(),8,0.05,500)
    model,inliers,error=ransac.solve()
    ret=ret[inliers]
    idx1=idx1[inliers]
    idx2=idx2[inliers]
    

    pose=extractRt(model)





    point_p,point_c=sanity_check_triangulation(f1, f2, idx1, idx2, pose)

    # P1=np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    # # P2 = np.linalg.inv(pose)[:3, :4]
    # P2 = pose[:3, :4]
    # pts4d=cv2.triangulatePoints(P1,P2,ret[:,1,:2].T,ret[:,0,:2].T)
    
    
    # # pts4d=triangulate(np.eye(4),pose,ret[:,1,:2],ret[:,0,:2])
    # # pts4d=pts4d.T
    # # print(pts4d)
    # pts4d/=pts4d[3,:]


    # point_p=np.dot(P1,pts4d)
    # point_c=np.dot(P2,pts4d)
    
    
    # point_p/=point_p[2,:]
    # point_c/=point_c[2,:]
    # # print(point_p.T)

    # return idx1-> current frame
    # return idx2-> previous frame
    return point_p,point_c,idx1,idx2
    # print(pts4d)



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
                ret.append((kp1,kp2))

    assert len(ret)>=3
    ret=np.array(ret).astype(np.float32)


    idx1=np.array(idx1)
    idx2=np.array(idx2)

    # ret[:,0,:2]=normalize(ret[:,0,:2],f1.Kinv)
    # ret[:,1,:2]=normalize(ret[:,1,:2],f2.Kinv)
    
    ransac=RANSAC(ret,Transformation(),3,0.05,100)
    model,inliers,error=ransac.solve()
    # print(error)
    idx1=idx1[inliers]
    idx2=idx2[inliers]

    pose=extractRt(model)

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

        

        depth_Z= depth.reshape((-1,1))    
        depth_Z = np.float32(depth_Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        n = 10 # this need to be tune
        ret,label,center=cv2.kmeans(depth_Z,n,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        label_img=label.reshape(depth.shape)
        
        self.pts,self.des,self.label=extract(img,depth,label_img)

    
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
        







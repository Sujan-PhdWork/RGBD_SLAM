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


def extract(img,depth):
    
    orb=cv2.ORB_create(100)
    feats=cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
    # print(feats)
    kps=[cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in feats]
    
    # _size in numpy<1.75
    
    kps,des=orb.compute(img,kps)
    # print(des.shape)
    modified_kps=[]
    modified_des=[]        
    
    for i,kp in enumerate(kps):
        u,v=map(lambda x: int(x),kp.pt)
        z=depth[v,u]
        if z!=0:
            modified_kps.append(kp)
            modified_des.append(des[i])   
    
    modified_des=np.array(modified_des)

    return np.array([(kp.pt[0],kp.pt[1],
                      depth[int(round(kp.pt[1])),int(round(kp.pt[0]))]) for kp in modified_kps]),modified_des


def match(f1,f2):
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.knnMatch(f1.des,f2.des,k=2)
    
    #low's ratio test
    ret=[]
    idx1,idx2=[],[]
    pose=None

    for m,n in matches:
        if m.distance <0.75*n.distance:
            if m.distance < 32:
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
    
    ransac=RANSAC(ret,Transformation(),8,0.01,500)
    model,inliers,error=ransac.solve()

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

        pts,self.des=extract(img,depth)
        # labels = kmeans_loaded.predict(self.des)
        # self.hist, _ = np.histogram(labels, bins=kmeans_loaded.n_clusters)
        self.Ihist=None
        #pts is 3d points unlormalize point

        self.K=K

        
        self.Kinv=np.linalg.inv(K)

        self.kps=pts.copy()

        self.kps[:,:2]=normalize(self.kps[:,:2],self.Kinv)
        
        #kps is 3d  normalize point 
        self.kps[:,2]=self.kps[:,2]/5000.0 #factor 
        
        self.kps[:,0]=self.kps[:,0]*self.kps[:,2]
        self.kps[:,1]=self.kps[:,1]*self.kps[:,2]

        self.pose=IRt
        self.Rpose=IRt
        self.id=len(mapp.frames)
        self.isKey=False
        







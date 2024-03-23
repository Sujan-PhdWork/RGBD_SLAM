import cv2
import numpy as np
from transformation import *
from ransac import *
from utils import normalize
import joblib

model_filename = 'BoW/kmeans_model.joblib'
kmeans_loaded = joblib.load(model_filename)


IRt=np.eye(4)

def extractRt(model):
    R=model.params['R']
    t=model.params['t']
    c=model.params['c']

    pose=np.eye(4)
    pose[:3,:3]=c*R
    pose[:3,3]=t.reshape(3)

    return pose


def extract(img,depth):
    
    orb=cv2.ORB_create(100)
    feats=cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
    kps=[cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in feats]
    kps,des=orb.compute(img,kps) 

    modified_des=[]
    modified_kps=[]

    for i,kp in enumerate(kps):
        u,v=map(lambda x: int(x),kp.pt)
        z=depth[int(v),int(u)]
        if z>0.05:
            modified_kps.append([u,v,z])
            modified_des.append(des[i])   
    
    modified_des=np.array(modified_des)
    modified_kps=np.array(modified_kps,dtype=np.uint16)
    # return modified_kps,modified_des   
    return modified_kps,modified_des


def match(f2,f1):
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.knnMatch(f2.des,f1.des,k=2)
    
    #low's ratio test
    ret=[]
    idx1,idx2=[],[]
    pose=None
    vmatch21=[]


    for m,n in matches:
        if m.distance <0.75*n.distance:
            idx2=m.queryIdx
            idx1=m.trainIdx
            vmatch21.append((idx2,idx1))
            

    assert len(vmatch21)>=3
    vmatch21=np.array(vmatch21).astype(np.uint16)

    # idx1=np.array(idx1)
    # idx2=np.array(idx2)

    # ret[:,0,:2]=normalize(ret[:,0,:2],f1.Kinv)
    # ret[:,1,:2]=normalize(ret[:,1,:2],f2.Kinv)
    
    ransac=RANSAC(f1,f2,vmatch21,Transformation(),10,0.05,200)
    model,inliers,error=ransac.solve()

    vmatch21=vmatch21[inliers]
    # idx1=idx1[inliers]
    # idx2=idx2[inliers]

    pose=extractRt(model)
    # pose=np.eye(4)

    return vmatch21,pose 


class Frame(object):
    def __init__(self,mapp,img,depth,K):
        
        
        self.kups,self.des=extract(img,depth)
        # labels = kmeans_loaded.predict(self.des)
        # self.hist, _ = np.histogram(labels, bins=kmeans_loaded.n_clusters)
        self.Ihist=None
        #pts is 3d points unlormalize point

        self.K=K
        self.Kinv=np.linalg.inv(K)

        self.kps=self.kups.astype(np.float64)

        self.kps[:,:2]=normalize(self.kps[:,:2],self.Kinv)
        #kps is 3d  normalize point 
        self.kps[:,2]=self.kps[:,2]/5000.0 #factor 
        
        self.kps[:,0]=self.kps[:,0]*self.kps[:,2]
        self.kps[:,1]=self.kps[:,1]*self.kps[:,2]

        #index of the points
        self.matchPts=np.empty(self.kps.shape[0],dtype=np.int32)
        self.matchPts.fill(-1)
        
        
        self.pose=IRt
        self.Rpose=IRt
        self.id=len(mapp.frames)
        mapp.frames.append(self)

import cv2
import numpy as np
from transformation import *
from ransac import *
from utils import normalize


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

    return np.array([(kp.pt[0],kp.pt[1],
                      depth[int(round(kp.pt[1])),int(round(kp.pt[0]))]) for kp in kps]),des


def match(f1,f2):
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.knnMatch(f1.des,f2.des,k=2)
    
    #low's ratio test
    ret=[]
    idx1,idx2=[],[]
    pose=None

    for m,n in matches:
        if m.distance <0.75*n.distance:

            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)

            # idx1-> id of the keypoint in previous frame
            # idx2-> id of the keypoint in current frame

            kp1=f1.kps[m.queryIdx]
            kp2=f2.kps[m.trainIdx]

            # kp1[idx]-> the keypoint in previous frame with id =idx            
            ret.append((kp1,kp2))

    assert len(ret)>=3
    ret=np.array(ret)

    idx1=np.array(idx1)
    idx2=np.array(idx2)

    # ret[:,0,:2]=normalize(ret[:,0,:2],f1.Kinv)
    # ret[:,1,:2]=normalize(ret[:,1,:2],f2.Kinv)
    
    ransac=RANSAC(ret,Transformation(),2,3,100)
    model,inliers,error=ransac.solve()

    idx1=idx1[inliers]
    idx2=idx2[inliers]

    pose=extractRt(model)

    return idx1,idx2,pose   

class Frame(object):
    def __init__(self,mapp,img,depth,K):
        
        
        pts,self.des=extract(img,depth)

        #pts is 3d points unlormalize point

        self.K=K
        self.Kinv=np.linalg.inv(K)

        self.kps=pts.copy()

        self.kps[:,:2]=normalize(self.kps[:,:2],self.Kinv)
        #kps is 3d  normalize point 

        self.pts=[None]*len(self.kps)

        self.pose=IRt
        self.id=len(mapp.frames)
        mapp.frames.append(self)


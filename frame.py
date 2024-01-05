import cv2
import numpy as np
from transformation import *
from ransac import *
from utils import normalize


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
    pose=None
    for m,n in matches:
        if m.distance <0.75*n.distance:
            kp1=f1.pts[m.queryIdx]
            kp2=f2.pts[m.trainIdx]
            ret.append((kp1,kp2))

    assert len(ret)>=3
    ret=np.array(ret)

    ret[:,0,:2]=normalize(ret[:,0,:2],f1.Kinv)
    ret[:,1,:2]=normalize(ret[:,1,:2],f2.Kinv)
    
    ransac=RANSAC(ret,Transformation(),3,3,500)
    model,inlier,error=ransac.solve()

    ret=np.array(ret)

    ret=ret[inlier]
    pose=extractRt(model)

    return ret,pose   

class Frame(object):
    def __init__(self,img,depth,K):
        self.pts,self.des=extract(img,depth)
        self.K=K
        self.Kinv=np.linalg.inv(K)

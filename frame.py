import cv2
import numpy as np
from transformation import *
from ransac import *

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
    
    orb=cv2.ORB_create(500)
    feats=cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=10)
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
            

            #around indices
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)

            kp1=f1.kps[m.queryIdx]
            kp2=f2.kps[m.trainIdx]
            ret.append((kp1,kp2))

    assert len(ret)>=8
    ret=np.array(ret)

    idx1=np.array(idx1)
    idx2=np.array(idx2)
    
    ransac=RANSAC(ret,Transformation(),3,0.5,500)
    model,inliers,error=ransac.solve()

    ret=np.array(ret)

    # ret=ret[inlier]
    idx1=idx1[inliers]
    idx2=idx2[inliers]
    pose=extractRt(model)
        
    
    return idx1,idx2,pose


class Frame(object):
    def __init__(self,mapp,img,depth):
        self.kps,self.des=extract(img,depth)
        self.pts=[None]*len(self.kps)
        self.pose=IRt
        self.id=len(mapp.frames)
        mapp.frames.append(self)
        self.K=np.eye(3)


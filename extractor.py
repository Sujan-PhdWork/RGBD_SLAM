import cv2
import numpy as np
from transformation import *
from ransac import *



def add_ones(x):
    return np.concatenate([x,np.ones((x.shape[0],1))],axis=1)


class Extractor():
    def __init__(self,K):
        self.orb=cv2.ORB_create(100)
        self.bf=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last=None
        self.K=K
        self.Kinv=np.linalg.inv(self.K)

    def normalize(self,pts):
        return np.dot(self.Kinv,add_ones(pts).T).T[:,0:2]
    
    def denormalize(self,pt):
        ret=np.dot(self.K,np.array([pt[0],pt[1],1.0]))
        return int(round(ret[0])),int(round(ret[1])),pt[2]

    def extract(self,img,depth):
        #detection
        feats=cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
        
        #extraction
        kps=[cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in feats]
        kps,des=self.orb.compute(img,kps)
        matches=None
        #matching
        ret=[]
        if self.last is not None:
            matches=self.bf.knnMatch(des,self.last['des'],k=2)
            for m,n in matches:
                if m.distance<0.75*n.distance:
                    kp1=kps[m.queryIdx].pt

                    #projected 2d point to 3d
                    u1,v1=map(lambda x: int(round(x)),kp1)
                    Z1=depth[v1,u1]
                    kp1=(u1,v1,Z1)


                    #projected 2d point to 3d
                    kp2=self.last['kps'][m.trainIdx].pt
                    u2,v2=map(lambda x: int(round(x)),kp2)
                    Z2=depth[v2,u2]
                    kp2=(u2,v2,Z2)
                    ret.append((kp1,kp2))

        

        if len(ret)>0:
            ret=np.array(ret)
            ret[:,0,:2]=self.normalize(ret[:,0,:2])
            ret[:,1,:2]=self.normalize(ret[:,1,:2])

            ransac=RANSAC(ret,Transformation(),3,3,500)
            model,inlier,error=ransac.solve()

            # ret=np.array(ret)

            ret=ret[inlier]




        self.last={'kps':kps,'des':des}
        return ret
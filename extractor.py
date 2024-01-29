import cv2
import numpy as np

class Extractor():
    def __init__(self):
        self.orb=cv2.ORB_create(100)
        # self.bf=cv2.BFMatcher()
        self.last=None
    def extract(self,img,depth):
        #detection
        feats=cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=10)
        
        #extraction
        kps=[cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in feats]
        
        
        

        kps,des=self.orb.compute(img,kps)
        
        kps=list(kps)
        des=list(des)

        modified_kps=[]
        modified_des=[]        
        for i,kp in enumerate(kps):
            u,v=map(lambda x: int(x),kp.pt)
            z=depth[v,u]
            if z==0:
                modified_kps.append(kp)
                modified_des.append(des[i])            

        
        # matches=None
        # #matching
        # if self.last is not None:
        #      matches=self.bf.match(des,self.last['des'])
        self.last={'kps':kps,'des':des}
        return modified_kps,modified_des
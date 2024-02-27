
import cv2
from frame import match
from pointmap import EDGE
import numpy as np

def global_localization(mapp,frame,th):

        # number of features in current frame  
        N=len(frame.des)
        
        dcos_list=[]
        for k in mapp.keyframes:
            # print(f1.id-f.id)
            # if (f1.id-f.id)<20:
            #     continue
            
            
            
            brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
            matches1 = brute_force.match(k.frame.des,frame.des)
            
            #number of matched features
            
            N1=len(matches1)
            
            #
            print(" : ",(N1/N))
            # print("ratio of the matched features: ",N1/N)

            # matches2 = brute_force.match(f.des,f2.des)
            # N2=len(matches2)

            # matches3 = brute_force.match(f.des,f3.des)
            # N3=len(matches3)

            # avg maches
            # N=(N1+N2+N3)/3.0

            if (N1/N)>=th:
                # print(f1.id,'::',k.frame.id)
                _,_,pose=match(frame,k.frame)
                # pose=GICP(f,f1)
                frame.pose=np.dot(pose,k.frame.pose)
                # EDGE(mapp,k.frame.id,frame.id,pose,3)
                print(k.frame.id," :..................... ", 'Loop closing')
                return True
        print("unable to find Keyframe")
        return False
    
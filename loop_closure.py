import numpy as np
from frame import match
import random
from threading import Thread,Lock,Event
from pointmap import EDGE
from time import sleep
import cv2
from GICP_test import GICP



class LoopThread(Thread):
    def __init__(self,mapp,lock,th):
        Thread.__init__(self)
        self.mapp=mapp
        self.lock=lock
        self.daemon=True
        self.th=th
        self.event=Event()

    
    def run(self):

        while True:
            # if self.event.isSet():
            
            with self.lock:
                if self.event.isSet():
                    if len(self.mapp.frames)>1:
                    
                        loop_closure(self.mapp,self.th)
                        # local_mapping(self.submap)
                        # self.event.clear()
                        # self.event.clear()    
                        sleep(3)
                else:
                    self.event.wait()

    


def loop_closure(mapp,th):
    if len(mapp.frames)<30:
        return


    sampled_frames=random.sample(mapp.frames[:-1], 28)
    
    sampled_frames.append(mapp.frames[-1])
    sampled_frames.append(mapp.frames[-2])
    sampled_frames.append(mapp.frames[-3])
    
    # TF_IDF(sampled_frames)

    f1=sampled_frames[-1]
    f2=sampled_frames[-2]
    f3=sampled_frames[-3]

    # number of features in current frame  
    N=len(f1.des)
    
    dcos_list=[]
    for f in sampled_frames[:-3]:
        # print(f1.id-f.id)
        if (f1.id-f.id)<20:
            continue
        
        
        
        brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        matches1 = brute_force.match(f.des,f1.des)
        
        #number of matched features
         
        N1=len(matches1)
        
        #
        # print(f.id," : ",f1.id)
        # print("ratio of the matched features: ",N1/N)

        # matches2 = brute_force.match(f.des,f2.des)
        # N2=len(matches2)

        # matches3 = brute_force.match(f.des,f3.des)
        # N3=len(matches3)

        # avg maches
        # N=(N1+N2+N3)/3.0

        if (N1/N)>=th:
            _,_,pose=match(f1,f)
            # pose=GICP(f,f1)
            EDGE(mapp,f.id,f1.id,pose,0.6)
            print(f.id," : ", (N1/N))
        
        
        


        
    # print("min_dcos_list",min(dcos_list),len(mapp.keyframes))




class Loop_Thread(object):
    def __init__(self):
        pass
    def create_Thread(self,mapp,th):
        lock=Lock()
        self.lc=LoopThread(mapp,lock,th)
        self.lc.start()
        self.lc.event.clear()


    
    


    
  
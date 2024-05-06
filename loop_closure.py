import numpy as np
from frame import match
import random
from threading import Thread,Lock,Event
from pointmap import EDGE
from time import sleep
import cv2
from GICP_test import GICP
import copy



class LoopThread(Thread):
    def __init__(self,mapp,lock,th):
        Thread.__init__(self)
        self.mapp=mapp
        self.lock=lock
        self.daemon=True
        self.th=th
        self.event=Event()
        self.nKframes=0

    def loop_closure(self,keyframes,th):
        if len(keyframes)<3:
            return

        # T_Keyframes=copy.deepcopy(keyframes)    
        
        if len(keyframes)>20:
            sampled_Keyframes=random.sample(keyframes[:-1],20)
        
        else:
            sampled_Keyframes=random.sample(keyframes[:-1], round(len(keyframes)/3))
        # sampled_frames.append(mapp.frames[-2])
        # sampled_frames.append(mapp.frames[-3])
        
        # TF_IDF(sampled_frames)

        f1=keyframes[-1].frame

        # number of features in current frame  
        N=len(f1.des)
        
        dcos_list=[]
        for k in sampled_Keyframes[:-1]:
            # print(f1.id-f.id)
            # if (f1.id-f.id)<20:
            #     continue
            
            
            
            brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
            matches1 = brute_force.match(k.frame.des,f1.des)
            
            #number of matched features
            
            N1=len(matches1)
            
            #
            # print(" : ",(N1/N))
            # print("ratio of the matched features: ",N1/N)

            # matches2 = brute_force.match(f.des,f2.des)
            # N2=len(matches2)

            # matches3 = brute_force.match(f.des,f3.des)
            # N3=len(matches3)

            # avg maches
            # N=(N1+N2+N3)/3.0

            if (N1/N)>=th:
                # print(f1.id,'::',k.frame.id)
                _,_,pose=match_by_segmentation(f1,k.frame)
                # pose=GICP(f,f1)
                with self.lock:
                    EDGE(self.mapp,k.frame.id,f1.id,pose,3)
                    print(k.frame.id," :..................... ", 'Loop closing')
                    return
                
        # del T_Keyframes   



    def run(self):

        while True:
            # if self.event.isSet():
            
            
            if self.event.isSet():
                with self.lock:
                    tkeys=copy.deepcopy(self.mapp.keyframes)
                if (len(tkeys)-self.nKframes)>0: 
                    self.nKframes=len(tkeys)
                    if self.nKframes>1:
                        # print('hello')
                        self.loop_closure(tkeys,self.th)
                        del tkeys
                        # local_mapping(self.submap)
                        # self.event.clear()
                        # self.event.clear()    
                        sleep(1)
                    else:
                        del tkeys
                        sleep(1)
                else:
                        del tkeys
                        sleep(1)
            else:
                self.event.wait()

    


        
        


        
    # print("min_dcos_list",min(dcos_list),len(mapp.keyframes))




class Loop_Thread(object):
    def __init__(self):
        pass
    def create_Thread(self,mapp,th):
        lock=Lock()
        self.lc=LoopThread(mapp,lock,th)
        self.lc.start()
        self.lc.event.clear()


    
    


    
  
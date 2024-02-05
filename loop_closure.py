import numpy as np
from frame import match
import random
from threading import Thread,Lock,Event
from pointmap import EDGE
from time import sleep
import cv2



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
    if len(mapp.frames)<20:
        return


    sampled_frames=sampled_frames=random.sample(mapp.frames[:-1], 13)
    
    sampled_frames.append(mapp.frames[-1])
    sampled_frames.append(mapp.frames[-2])
    sampled_frames.append(mapp.frames[-3])
    
    # TF_IDF(sampled_frames)

    f1=sampled_frames[-1]
    f2=sampled_frames[-2]
    f3=sampled_frames[-3]


    dcos_list=[]
    for f in sampled_frames[:-3]:
        print(f1.id-f.id)
        if (f1.id-f.id)<20:
            continue
        brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        no_of_matches1 = brute_force.match(f.des,f1.des)
        no_of_matches2 = brute_force.match(f.des,f2.des)
        no_of_matches3 = brute_force.match(f.des,f3.des)
        print(f.id,": ",f1.id," no of matches :",len(no_of_matches1))
        print(f.id,": ",f2.id," no of matches :",len(no_of_matches2))
        print(f.id,": ",f3.id," no of matches :",len(no_of_matches3))  
        print(" ")    


        
    # print("min_dcos_list",min(dcos_list),len(mapp.keyframes))




class Loop_Thread(object):
    def __init__(self):
        pass
    def create_Thread(self,mapp,th):
        lock=Lock()
        self.lc=LoopThread(mapp,lock,th)
        self.lc.start()


    
    


    
  
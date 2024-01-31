# It will check if a frame is minimum match to keyframe then that frame is detected as keyframe 

from frame import match
from threading import Thread,Lock,Event
from pointmap import Map
import numpy as np
from  local_mapping import local_mapping  



class KeyframeThread(Thread):
    def __init__(self,mapp,lock,th):
        Thread.__init__(self)
        self.mapp=mapp
        self.Keyframe=None
        self.lock=lock
        self.daemon=True
        self.th=th
        
        # self.event=Event()

    def Keyframe_detection(self):
        submap=Map()
        Rpose=np.eye(4)
        keyframe=self.mapp.keyframes[-1]
        keyid=self.mapp.keyframes[-1].id
        for f in self.mapp.frames[keyframe.id:]:
            print("second")
            if f.id==keyid:
                submap.frames.append(f)
                continue

            idx1,idx2,pose=match(f,keyframe)
            
            # This is the relative position from the keyframe
            # Rpose=np.dot(pose,Rpose)
            # f.pose=Rpose.copy()

            
            
            f.keyid=keyid
            
        
        
            if len(idx1)<self.th:
                
                
                
                #putting key frame pose as identity
                # f.pose=np.eye(4)
                local_mapping(submap)
                self.mapp.keyframes.append(f)
                break
            elif len(idx1)>self.th:
                submap.frames.append(f)
        
        # sending the submap to calculate pose 
        # or we can create the local map
               
        
            
    def run(self):

        while True:
            with self.lock:
                if len(self.mapp.frames)>1:
                    self.Keyframe_detection()
        
        


    





class Keyframes(object):
    def __init__(self):
        pass
    def create_Thread(self,mapp,th):
        lock=Lock()
        self.kf=KeyframeThread(mapp,lock,th)
        self.kf.start()
        
        # self.kf.event.set()

    



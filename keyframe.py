# It will check if a frame is minimum match to keyframe then that frame is detected as keyframe 

from frame import match
from threading import Thread,Lock,Event
from pointmap import Map
import numpy as np
from  local_mapping import local_mapping  
from time import sleep



class KeyframeThread(Thread):
    def __init__(self,mapp,lock,th):
        Thread.__init__(self)
        self.mapp=mapp
        self.Keyframe=None
        self.lock=lock
        self.daemon=True
        self.th=th
        self.event=Event()
        self.submap=Map()
        
        # self.event=Event()

    def Keyframe_detection(self):
        submap=Map()
        Rpose=np.eye(4)
        keyframe=self.mapp.keyframes[-1]
        keyid=self.mapp.keyframes[-1].id

        for f in self.mapp.frames[keyframe.id:]:
            if f.id==keyid:
                f.keyid=keyid
                submap.frames.append(f)
                continue

            idx1,idx2,pose=match(f,keyframe)

            f.keyid=keyid

                
            if len(idx1)<self.th:
                self.submap=submap
                # local_mapping(submap)
                # submap.frames.insert(0,keyframe)
                # print("new_key frame is added ")
                self.mapp.keyframes.append(f)
            elif len(idx1)>self.th:
                submap.frames.append(f)
               
        
            
    def run(self):

        while True:
            if self.event.isSet():
                with self.lock:
                    if len(self.mapp.frames)>1:
                        self.Keyframe_detection()
                        self.event.clear()
            else:
                self.event.wait()

        
        


    





class Keyframes(object):
    def __init__(self):
        pass
    def create_Thread(self,mapp,th):
        lock=Lock()
        self.kf=KeyframeThread(mapp,lock,th)
        self.kf.start()
        
        # self.kf.event.set()

    



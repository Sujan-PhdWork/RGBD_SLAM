# It will check if a frame is minimum match to keyframe then that frame is detected as keyframe 

from frame import match
from threading import Thread,Lock,Event
from pointmap import Map




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
        for f in self.mapp.frames[self.mapp.keyframes[-1].id:]:
            keyid=self.mapp.keyframes[-1].id
            idx1,idx2,pose=match(f,self.mapp.keyframes[-1])
            f.keyid=keyid
            
            submap.frames.append(f)
            if len(idx1)<self.th:
                local_frames=[f.id for f in submap.frames]
                print(local_frames)
                self.mapp.keyframes.append(f)
                break
    
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

    



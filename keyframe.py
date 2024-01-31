# It will check if a frame is minimum match to keyframe then that frame is detected as keyframe 

from frame import match
from threading import Thread,Lock,Event
from pointmap import Map




class KeyframeThread(Thread):
    def __init__(self,mapp,lock):
        Thread.__init__(self)
        self.mapp=mapp
        self.Keyframe=None
        self.lock=lock
        self.daemon=True
        
        # self.event=Event()

    
    def run(self):

        while True:
            with self.lock:
                if len(self.mapp.frames)>1:
                    Keyframe_detection(self.mapp)
        
        

def Keyframe_detection(mapp,th=300):
    for f in mapp.frames[mapp.keyframes[-1].id:]:
        idx1,idx2,pose=match(f,mapp.keyframes[-1])
        if len(idx1)<th:
            # print(f.id)
            mapp.keyframes.append(f)
            break





class Keyframes(object):
    def __init__(self):
        pass
    def create_Thread(self,mapp):
        lock=Lock()
        self.kf=KeyframeThread(mapp,lock)
        self.kf.start()
        
        # self.kf.event.set()

    



# It will check if a frame is minimum match to keyframe then that frame is detected as keyframe 

from frame import match
from threading import Thread,Lock


class KeyframeThread(Thread):
    def __init__(self,mapp,lock):
        Thread.__init__(self)
        self.mapp=mapp
        self.value=None
        self.lock=lock
    
    def run(self):
        with self.lock:
            self.value=self.mapp.frames[-1]


def Keyframe_detection(mapp):
    lock=Lock()
    kf=KeyframeThread(mapp,lock)
    kf.start()
    return(kf.value)

    



import numpy as np

class Keyframe(object):
    def __init__(self,frame):
        self.id=frame.id
        self.frames=[frame]
        self.frame=frame
        # self.pose=np.eye(4)
        self.nmpts=None
        pass
    def add_frames(self,f):
        self.frames.append(f)
    def update_frames(self,Rpose):
        
        for f in self.frames:
            f.pose=np.dot(Rpose,f.pose)
        

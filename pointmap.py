import numpy as np
import pangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue
from viz import Disp_map


class Map(object):
    def __init__(self):
        self.frames=[]
        self.points=[]
        
        self.q=None
        self.Dmap=Disp_map()
            
    
    def create_viewer(self):
        self.q=Queue()
        self.p=Process(target=self.Dmap.viewer_thread,args=(self.q,))
        self.p.demon=True
        self.p.start()

    def display(self):
        poses,R_poses=[],[]
        for f in self.frames:
            poses.append(f.pose)
            R_poses.append(f.Rpose)
        
        self.q.put((np.array(poses),np.array(R_poses)))
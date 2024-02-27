import numpy as np
from multiprocessing import Process, Queue
from viz import Disp_map
import g2o


class Map(object):
    def __init__(self):
        self.frames=[]
        self.edges=[]
        
        self.keyframes=[]
        self.index=0
        
        self.q=None
        self.Dmap=Disp_map()


    def create_viewer(self):
        self.q=Queue()
        self.p=Process(target=self.Dmap.viewer_thread,args=(self.q,))
        self.p.demon=True
        self.p.start()

    def display(self):
        poses,R_poses=[],[]
        # for k in self.keyframes:
        for f in self.frames:
            poses.append(f.pose)
        for f in self.frames:
                R_poses.append(f.Rpose)
        # for e in self.edges:
        #     f1,f2=e.frames
        #     edges.append((f1.pose[:3,3],f2.pose[:3,3]))
        
        self.q.put((np.array(poses),np.array(R_poses)))



class EDGE(object):
    
    def __init__(self,mapp,id1,id2,pose,noise):
        
        
        f1=mapp.frames[id1]
        f2=mapp.frames[id2]

        print ("Adding edge between",f1.id,f2.id)
        self.frames=[f1,f2,noise]


        self.pose=pose
        self.id=len(mapp.edges)
        mapp.edges.append(self)
        
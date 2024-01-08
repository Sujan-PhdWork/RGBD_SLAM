import numpy as np
from multiprocessing import Process, Queue
from viz import Disp_map
import g2o


class Map(object):
    def __init__(self):
        self.frames=[]
        self.edges=[]
        
        self.q=None
        self.Dmap=Disp_map()
            
    
    def optimize(self):
        opt=g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)

        for f in self.frames:
            
            pose=f.pose
            v_se3=g2o.VertexSE3()
            v_se3.set_id(f.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(f.id==0)
            opt.add_vertex(v_se3)

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


class EDGE(object):
    
    def __init__(self,mapp,id1,id2,pose):
        
        f1=mapp.frames[id1]
        f2=mapp.frames[id2]

        self.frames=[f1,f2]
        self.pose=pose
        mapp.edges.append(self)
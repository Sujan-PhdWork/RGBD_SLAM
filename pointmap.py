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
    




    def optimize_process(self):
        self.p1=Process(target=self.optimize,args=())
        self.p1.demon=True
        self.p1.start()
    

    def optimize(self,n=20):
        opt=g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)

        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
        
        #
        

        for f in self.frames[-n:]:
            pose=f.pose
            v_se3=g2o.VertexSE3()
            v_se3.set_id(f.id)

            pcam=g2o.Isometry3d(pose[:3,:3], pose[:3,3])
            
            v_se3.set_estimate(pcam)
            v_se3.set_fixed(f.id==0)
            opt.add_vertex(v_se3)

        for edge in self.edges[-n:]:
            f1,f2=edge.frames
            pose=edge.pose
            Eg= g2o.EdgeSE3()
            Eg.set_vertex(0,opt.vertex(f1.id))
            Eg.set_vertex(1,opt.vertex(f2.id))
            scam=g2o.Isometry3d(pose[:3,:3], pose[:3,3])
            Eg.set_measurement(scam)
            Eg.set_information(5*np.eye(6))
            Eg.set_robust_kernel(robust_kernel)
            opt.add_edge(Eg)


        opt.initialize_optimization()
        opt.compute_active_errors()
        print('Initial chi2 =', opt.chi2())

        # opt.save('gicp.g2o')

        opt.set_verbose(False)
        opt.optimize(10)

        for f in self.frames[-n:]:
            est = opt.vertex(f.id).estimate()
            R = est.rotation().matrix()
            t = est.translation()
            ret=np.eye(4)
            ret[:3,:3]=R
            ret[:3,3]=t
            f.pose = ret.copy()




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
        self.q


class EDGE(object):
    
    def __init__(self,mapp,id1,id2,pose):
        
        f1=mapp.frames[id1]
        f2=mapp.frames[id2]

        self.frames=[f1,f2]
        self.pose=pose
        mapp.edges.append(self)
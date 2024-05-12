from frame import match
from threading import Thread,Lock,Event
# from pointmap import Map
import numpy as np
import copy
import g2o
from GICP_test import GICP

# from  local_mapping import local_mapping  
from time import sleep


class FullMAP(Thread):
    def __init__(self,mapp,lock):
        Thread.__init__(self)
        self.mapp=mapp
        # self.Keyframe=None
        self.lock=lock
        self.daemon=True
        self.event=Event()
        self.nKframes=0
        # self.CheckNewKeyframe=False

    def optimize(self):
            self.opt=g2o.SparseOptimizer()
            solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
            solver = g2o.OptimizationAlgorithmLevenberg(solver)
            self.opt.set_algorithm(solver)

            robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
            
            #

            for k in self.tkeyframes:
                pose=k.frame.pose
                v_se3=g2o.VertexSE3()
                v_se3.set_id(k.frame.id)
                pcam=g2o.Isometry3d(pose[:3,:3], pose[:3,3])
                
                v_se3.set_estimate(pcam)
                v_se3.set_fixed(k.frame.id==0)
                # print("In Full_map:",k.frame.id,len(self.tkeyframes))
                self.opt.add_vertex(v_se3)

            for edge in self.edges:
                f1,f2,noise=edge.frames
                pose=edge.pose
                Eg= g2o.EdgeSE3()
                # print("In Full_map:",f1.id,f2.id)
                Eg.set_vertex(0,self.opt.vertex(f1.id))
                Eg.set_vertex(1,self.opt.vertex(f2.id))
                scam=g2o.Isometry3d(pose[:3,:3], pose[:3,3])
                Eg.set_measurement(scam)
                Eg.set_information(noise*np.eye(6))
                # Eg.set_robust_kernel(robust_kernel)
                self.opt.add_edge(Eg)


            self.opt.initialize_optimization()
            self.opt.compute_active_errors()
            # print('Initial chi2 =', opt.chi2())

            # opt.save('gicp.g2o')

            # self.opt.set_verbose(True)
            self.opt.optimize(10)

            # Mistake:: 
            with self.lock:
                for k in self.mapp.keyframes:
                    # print(self.opt.vertex(k.frame.id))# is None:
                        #print('Your are not suppose to add now')
                        #continue


                    # print(k.frame.id)
                    est = self.opt.vertex(k.frame.id).estimate()
                    ret=np.eye(4)
                    ret[:3,:3]=est.rotation().matrix()
                    ret[:3,3]=t = est.translation()
                    # print(t)
                    R_pose=np.dot(ret,np.linalg.inv(k.frame.pose))
                    k.update_frames(R_pose)
                    k.frame.pose = ret.copy()
    
    def run(self):

             
        while True:
            if self.event.is_set():
                # sleep(3)
                with self.lock:

                    self.tkeyframes=copy.deepcopy(self.mapp.keyframes)
                    self.edges=copy.deepcopy(self.mapp.edges)

                
                # print(1)
                if len(self.tkeyframes)==1:
                    print('First key frame')
                    self.event.clear()
                    continue
                print('Full optimization...')
                self.optimize()
                del self.tkeyframes
                del self.opt
                # self.Keyframe.update_frames()
                # sleep(0.5)
                self.event.clear()
            
            
            else:
                self.event.wait()


class FulllMap_Thread(object):
        def __init__(self):
            pass
        def create_Thread(self,mapp):
            lock=Lock()
            self.fm=FullMAP(mapp,lock)
            self.fm.start()
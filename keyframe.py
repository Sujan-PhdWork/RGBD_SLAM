# # It will check if a frame is minimum match to keyframe then that frame is detected as keyframe 

from frame import match
from threading import Thread,Lock,Event
# from pointmap import Map
import numpy as np
import copy
import g2o
from GICP_test import GICP

# from  local_mapping import local_mapping  
from time import sleep



class LocalMAP(Thread):
    def __init__(self,mapp,lock):
        Thread.__init__(self)
        
        self.mapp=mapp
        # self.Keyframe=None
        self.lock=lock
        self.daemon=True
        self.Acceptance_flag=True
        self.event=Event()
        # self.CheckNewKeyframe=False
        self.NewKeyframes=[]
    



    def SetAcceptKeyFrames(self,flag):
        self.Acceptance_flag=flag
    
    def CheckNewKeyframe(self):
        return self.NewKeyframes
    
    
    def Process_newKeyframe(self):
        with self.lock:

            self.Keyframe=self.NewKeyframes[0]
            self.Current_Keyframe=copy.deepcopy(self.Keyframe)
            self.NewKeyframes.pop()
        
        
        self.optimize_initialize()
        print("enter")
        
        for i in range(len(self.Current_Keyframe.frames)):
            pose=self.Current_Keyframe.frames[i].pose
            v_se3=g2o.VertexSE3()
            v_se3.set_id(i)

            pcam=g2o.Isometry3d(pose[:3,:3], pose[:3,3])
            
            v_se3.set_estimate(pcam)
            v_se3.set_fixed(i==0)
            self.opt.add_vertex(v_se3)
            
        
        for i in range(len(self.Current_Keyframe.frames)-1):
            
            f1,f2=self.Current_Keyframe.frames[i],self.Current_Keyframe.frames[i+1]
            _,_,pose=match(f2,f1)
            # pose=GICP(self.Current_Keyframe.frames[i],self.Current_Keyframe.frames[i+1])
            # pose=edge.pose
            Eg= g2o.EdgeSE3()
            Eg.set_vertex(0,self.opt.vertex(i))
            Eg.set_vertex(1,self.opt.vertex(i+1))
            scam=g2o.Isometry3d(pose[:3,:3], pose[:3,3])
            Eg.set_measurement(scam)
            Eg.set_information(0.02*np.eye(6))
            Eg.set_robust_kernel(self.robust_kernel)
            self.opt.add_edge(Eg)
        
        



    def optimize_initialize(self):
        
        self.opt=g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.opt.set_algorithm(solver)
        self.robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    
    def optimize(self):
        self.opt.initialize_optimization()
        self.opt.compute_active_errors()
        self.opt.set_verbose(True)
        self.opt.optimize(100)
        self.update_pose()
        
        
    def update_pose(self):
        with self.lock:
            self.Keyframe=copy.deepcopy(self.Current_Keyframe)
            

        

                    
               
        
            
    def run(self):

        while True:
            # sleep(3)
            if self.CheckNewKeyframe():
                # print(1)
                self.SetAcceptKeyFrames(False)
                self.Process_newKeyframe()
                self.optimize()
                self.mapp.keyframes.append(self.Keyframe)
                self.SetAcceptKeyFrames(True)
            else:
                sleep(0.5)



class LocalMap_Thread(object):
        def __init__(self):
            pass
        def create_Thread(self,mapp):
            lock=Lock()
            self.lm=LocalMAP(mapp,lock)
            self.lm.start()
    


class Keyframe(object):
    def __init__(self,frame):
        self.id=frame.id
        self.frames=[]
        self.frame=frame
        self.pose=np.eye(4)
        self.nmpts=None
        pass
    def add_frames(self,f):
        self.frames.append(f)
    def update_frames(self):
        for f in self.frames:
            f.pose=np.dot(self.pose,f.pose)
        

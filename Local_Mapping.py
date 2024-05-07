from frame import match,match_by_segmentation
from threading import Thread,Lock,Event
# from pointmap import Map
import numpy as np
import copy
import g2o
from GICP_test import GICP
from Full_map import FulllMap_Thread


# from  local_mapping import local_mapping  
from time import sleep



class LocalMAP(Thread):
    def __init__(self,mapp,lock):
        Thread.__init__(self)
        

        
        self.mapp=mapp
        self.Full_MAP=FulllMap_Thread()
        self.Full_MAP.create_Thread(self.mapp)
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
        for k in self.NewKeyframes:
            # print('New Key: ',k.id,self.Acceptance_flag)
            return len(self.NewKeyframes)
    
    
    def Process_newKeyframe(self):
        with self.lock:
            self.Keyframe=self.NewKeyframes[0]
            self.Current_Keyframe=copy.deepcopy(self.NewKeyframes[0])
            self.NewKeyframes.pop()
        
        if len(self.Current_Keyframe.frames)==0:
            return
        self.optimize_initialize()
        print("Processing Local Mapping")
        # print(type(self.Current_Keyframe.frames[::-1]))
        #reverse the list
        # R_frames=copy.deepcopy()

        for f in self.Current_Keyframe.frames:
            pose=f.pose
            v_se3=g2o.VertexSE3()
            v_se3.set_id(f.id)

            pcam=g2o.Isometry3d(pose[:3,:3], pose[:3,3])
            
            v_se3.set_estimate(pcam)
            # print(f.isKey)
            v_se3.set_fixed(f.isKey)
            # print("in LocalMapping:",f.id)
            self.opt.add_vertex(v_se3)
            
        
        for i in range(len(self.Current_Keyframe.frames)-1):
            
            f1,f2=self.Current_Keyframe.frames[i],self.Current_Keyframe.frames[i+1]
            # print(self.Current_Keyframe.id,f1.id,f2.id)
            _,_,pose=match_by_segmentation(f2,f1)
            # pose=GICP(self.Current_Keyframe.frames[i],self.Current_Keyframe.frames[i+1])
            # pose=edge.pose
            Eg= g2o.EdgeSE3()
            Eg.set_vertex(0,self.opt.vertex(f2.id))
            Eg.set_vertex(1,self.opt.vertex(f1.id))
            scam=g2o.Isometry3d(pose[:3,:3], pose[:3,3])
            Eg.set_measurement(scam)
            Eg.set_information(5*np.eye(6))
            Eg.set_robust_kernel(self.robust_kernel)
            self.opt.add_edge(Eg)
        

        self.optimize()
        
        



    def optimize_initialize(self):
        
        self.opt=g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.opt.set_algorithm(solver)
        self.robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    
    def optimize(self):
        self.opt.initialize_optimization()
        self.opt.compute_active_errors()
        self.opt.set_verbose(False)
        self.opt.optimize(100)
        self.update_pose()
        del self.Current_Keyframe
        del self.opt
        
        
    def update_pose(self):
        with self.lock:
            for f in self.Keyframe.frames:
                # print( )
                est = self.opt.vertex(f.id).estimate()
                R = est.rotation().matrix()
                t = est.translation()
                ret=np.eye(4)
                ret[:3,:3]=R
                ret[:3,3]=t
                f.pose = ret.copy()
   
            # self.Keyframe=copy.deepcopy(self.Current_Keyframe)
            

        

                    
               
        
            
    def run(self):

        while True:
            # sleep(3)
            # print('before',self.CheckNewKeyframe())
            if self.CheckNewKeyframe():
                # print('after',self.CheckNewKeyframe())
                # print(1)
                # self.SetAcceptKeyFrames(False)
                self.Process_newKeyframe()
                # self.optimize()
                # self.Keyframe.update_frames()
                with self.lock:
                    print('Local map optimization...')
                    # print(self.Keyframe.id)
                    self.mapp.keyframes.append(self.Keyframe)
                    if self.Keyframe.id==0:
                        self.Full_MAP.fm.event.set()
                    else:
                        
                        while True:
                            if not self.Full_MAP.fm.event.is_set():
                                break

                        self.Full_MAP.fm.event.set()
                            
                self.SetAcceptKeyFrames(True)
                sleep(1)

            else:
                sleep(0.5)



class LocalMap_Thread(object):
        def __init__(self):
            pass
        def create_Thread(self,mapp):
            lock=Lock()
            self.lm=LocalMAP(mapp,lock)
            self.lm.start()
            

   
import numpy as np
import g2o
import random
from scipy.spatial import KDTree
import pcl.pcl_visualization
import pcl
from threading import Thread,Lock,Event
from pointmap import Map,EDGE
from time import sleep


#visual = pcl.pcl_visualization.CloudViewing()
viewer = pcl.pcl_visualization.PCLVisualizering()
class GICPThread(Thread):
    def __init__(self,mapp,lock):
        Thread.__init__(self)
        self.mapp=mapp
        self.lock=lock
        self.daemon=True
        self.event=Event()
        self.pose=np.eye(4)

    def run (self):
        while True:
            # if self.event.isSet():
            
            with self.lock:
                if self.event.isSet():
                    if len(self.mapp.frames)>1:
                        f_c=self.mapp.frames[-1]
                        f_p=self.mapp.frames[-2]
                        cloud1=f_c.cloud
                        cloud2=f_p.cloud

                        self.pose=GICP(cloud1,cloud2)
                        print("GICP",f_c.id,f_p.id)
                        EDGE(self.mapp,f_p.id,f_c.id,self.pose,0.5)

                        # self.event.clear()
                        # local_mapping(self.submap)
                        # self.event.clear()
                        # self.event.clear()    
                        # sleep(3)
                else:
                    self.event.wait()


def GICP(f_c,f_p):

    #cloud1: current frame cloud data
    #cloud2: previous frame cloud data

    cloud_p=f_p.cloud.copy()
    cloud_c=f_c.cloud.copy()
    
    
    poses=[np.eye(4),np.dot(f_c.pose,np.linalg.inv(f_p.pose))]
    
    pose=poses[1]
    t = pose[:3,3]
    R= pose[:3,:3]

    cloud_c=np.dot(R,cloud_c.T)+t.reshape(3,1)
    cloud_c=cloud_c.T

    
    pltcloud_p = pcl.PointCloud()
    pltcloud_p.from_array(cloud_p.astype(np.float32))
    viewer.AddPointCloud(pltcloud_p,b"cloud_prior")
    # viewer.AddPointCloud(pltcloud2,b"cloud2",1)
    


    j=1 
    pltcloud_c = pcl.PointCloud()
    pltcloud_c.from_array(cloud_c.astype(np.float32))
    pccolor = pcl.pcl_visualization.PointCloudColorHandleringCustom(pltcloud_c, 255,0,0)
    # viewer.AddPointCloud(pltcloud_c,bytes(str(j),encoding='utf8'))
    viewer.AddPointCloud_ColorHandler(pltcloud_c, pccolor, bytes(str(j),encoding='utf8'))
    viewer.SpinOnce()
    sleep(5)   
            
    # t_test=opt.vertex(1).estimate().t
    
    kdt=KDTree(cloud_p)
    for j in range(2,1000):
        # print(j)
        viewer.RemovePointCloud(bytes(str(j-1),encoding='utf8'),0)

        
        
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
        opt.set_algorithm(algorithm)

        for i in range(2):
            pose=poses[i]
            pcam = g2o.Isometry3d(pose[:3,:3],pose[:3,3])
            
            vc = g2o.VertexSE3()
            vc.set_id(i)
            vc.set_estimate(pcam)
            vc.set_fixed(i==0)
            
            opt.add_vertex(vc)

        idx=np.random.choice(cloud_c.shape[0], 10000, replace=False)    
        sampled_cloudC = cloud_c[idx,:]
        dist, indices = kdt.query(sampled_cloudC,k=1,p=2,distance_upper_bound=0.01)
        

        for i in range(sampled_cloudC.shape[0]):
            if dist[i]>2:
                continue
            elif dist[i]<2:
                meas = g2o.EdgeGICP()
                meas.pos0 = cloud_p[indices[i],:]
                meas.pos1 = sampled_cloudC[i,:]

                edge = g2o.Edge_V_V_GICP()
                edge.set_vertex(0, opt.vertex(0))
                edge.set_vertex(1, opt.vertex(1))
                edge.set_measurement(meas)
                edge.set_information(meas.prec0(0.01))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)


        opt.initialize_optimization()
        opt.compute_active_errors()
    #     # if verbose:
        print('GICP Initial chi2 =', opt.chi2())

        # opt.set_verbose(True)
        opt.optimize(10)
  
        vc = opt.vertex(1)
        
        T=np.eye(4)
        
        # ensuresing the orthonormal matrix
        # u and vh are both orthogonal matrices, meaning their transpose is their inverse.
        
        u,s,vh = np.linalg.svd(vc.estimate().R) 
        T[:3,:3]=u @ vh
        
        # print(np.linalg.det(vc.estimate().R))
        T[:3,3]=vc.estimate().t


        print('GICP',T)
        poses[1]=T

        cloud_c=np.dot(vc.estimate().R,cloud_c.T)+vc.estimate().t.reshape(3,1)
        cloud_c=cloud_c.T
        
        
        pltcloud3 = pcl.PointCloud()
        pltcloud3.from_array(cloud_c.astype(np.float32))
        
        pccolor = pcl.pcl_visualization.PointCloudColorHandleringCustom(pltcloud3, 255,0,0)
        # viewer.AddPointCloud(pltcloud_c,bytes(str(j),encoding='utf8'))
        viewer.AddPointCloud_ColorHandler(pltcloud3, pccolor, bytes(str(j),encoding='utf8'))

        # viewer.AddPointCloud(pltcloud3,bytes(str(j),encoding='utf8'))
        viewer.SpinOnce()
        sleep(0.1) 
        
   
    return pose


class GICP_Thread(object):
    def __init__(self):
        pass
    def create_Thread(self,mapp):
        lock=Lock()
        self.gc=GICPThread(mapp,lock)
        self.gc.start()
        self.gc.event.clear()

        








        


        





import numpy as np
import g2o

def PoseOptimization(f1,f2,idx1,idx2,Pose):

    K=f1.K
    kps1= f1.kps[idx1]
    pts1= f1.pts[idx1][:,:2]

    kps2= f2.kps[idx2]
    pts2= f2.pts[idx2][:,:2]


    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    

    focal_length = K[0,0]
    principal_point = (K[0,2], K[1,2])
    cam = g2o.CameraParameters(focal_length, principal_point, 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)    
    
    
    cam = g2o.Isometry3d(np.eye(4)[:3,:3], np.eye(4)[:3,3])
    vc=g2o.VertexSE3()
    vc.set_id(0)
    vc.set_fixed(True)
    vc.set_estimate(cam)
    optimizer.add_vertex(vc)


    cam = g2o.Isometry3d(Pose[:3,:3], Pose[:3,3])
    vc=g2o.VertexSE3()
    vc.set_id(1)
    vc.set_fixed(False)
    vc.set_estimate(cam)
    optimizer.add_vertex(vc)


    R=Pose[:3,:3]
    t=Pose[:3,3]
    
    kps1=np.dot(R,kps1.T)+t.reshape(3,1)
    kps1=kps1.T    

    N=kps1.shape[0]

    for i in range(N):
        vp=g2o.VertexSBAPointXYZ()
        vp.set_estimate(kps2[i])
        vp.set_id(i+2)
        vp.set_fixed(True)
        vp.set_marginalized(True)
        optimizer.add_vertex(vp)

        
        z1 = pts1[i]
        z2 = pts2[i]

        


        e=g2o.EdgeProjectXYZ2UV()
        e.set_vertex(0,vp)
        e.set_vertex(1,optimizer.vertex(0))
        e.set_measurement(z2)
        e.set_information(np.identity(2)*0.05)
        e.set_robust_kernel(g2o.RobustKernelHuber())
        e.set_parameter_id(0, 0)
        optimizer.add_edge(e)


        e=g2o.EdgeProjectXYZ2UV()
        e.set_vertex(0,vp)
        e.set_vertex(1,optimizer.vertex(1))
        e.set_measurement(z1)
        e.set_information(np.identity(2)*0.05)
        e.set_robust_kernel(g2o.RobustKernelHuber())
        e.set_parameter_id(0, 0)
        optimizer.add_edge(e)




    optimizer.initialize_optimization()
    print("before",optimizer.chi2())
    optimizer.set_verbose(True)
    optimizer.optimize(100)
    print("after",optimizer.chi2())
    # print(optimizer.vertex(1).estimate().t)

    # er=0
    # for i in range(N):
    #     vp = optimizer.vertex(i+2)
    #     error = vp.estimate() - kps1[i]
    #     er += np.sum(error**2)
    # print(er)
    # t=optimizer.vertex(1).estimate().inverse().t
    # R=optimizer.vertex(1).estimate().inverse().R

    t=optimizer.vertex(1).estimate().t
    R=optimizer.vertex(1).estimate().R

    pose=np.eye(4)
    pose[:3,:3]=R
    pose[:3,3]=t

    return pose    




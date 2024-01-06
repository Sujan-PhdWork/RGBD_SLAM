import numpy as np
import g2o

def gicp(mapp,verbose=False):
    

    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(algorithm)
    f1,f2=mapp.frames[-2],mapp.frames[-1]

    f_list=[f1,f2]


    for i in range(len(f_list)):
        f=f_list[i]
        pose=np.dot(np.linalg.inv(f1.pose),f.pose)
  
        t = pose[:3,3]
        R= pose[:3,:3]

        pcam = g2o.Isometry3d(R, t)
        
        vc = g2o.VertexSE3()
        vc.set_id(i)
        vc.set_estimate(pcam)
        vc.set_fixed(i==0)
        
        opt.add_vertex(vc)
        
    trans0 = opt.vertex(0).estimate().inverse()
    trans1 = opt.vertex(1).estimate().inverse()

    # # set up point matches

    pts1=mapp.frames[-2].pts # points on previous frame
    pts2= mapp.frames[-1].pts # points on current frame

    for i in range(len(pts1)):
    
    #     # calculate the relative 3d position of the point
        pt0 = trans0 * pts1[i]
        pt1 = trans1 * pts2[i]

        meas = g2o.EdgeGICP()
        meas.pos0 = pt0
        meas.pos1 = pt1

        edge = g2o.Edge_V_V_GICP()
        edge.set_vertex(0, opt.vertex(0))
        edge.set_vertex(1, opt.vertex(1))
        edge.set_measurement(meas)
        edge.set_information(meas.prec0(0.01))
        opt.add_edge(edge)


    opt.initialize_optimization()
    opt.compute_active_errors()
    if verbose:
        print('GICP Initial chi2 =', opt.chi2())

    R_pose=np.eye(4)

    # print(opt.vertex(1).estimate().R)
    R_pose[:3,:3]=opt.vertex(1).estimate().R
    R_pose[:3,3]=opt.vertex(1).estimate().t

    return R_pose
        


        




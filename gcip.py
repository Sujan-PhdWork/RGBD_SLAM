import numpy as np
import g2o

def gcip(frames,pts):
    
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(algorithm)
    f1,f2=frames

    for i in range(len(frames)):
        f=frames[i]
        pose=f.pose
        
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
    # set up point matches


    for i in len(pts[0].pts):
        # calculate the relative 3d position of the point
        pt0 = pts[0].pts[i]
        pt1 = pts[0].pti
        


        





import numpy as np
import g2o



def main():
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)



    for i in range(5):
        t = np.array([0, 0, i])
        cam = g2o.Isometry3d(np.identity(3), t)

        vc = g2o.VertexSE3()
        vc.set_id(i)
        vc.set_estimate(cam)
        if i == 0:
            vc.set_fixed(True)
        optimizer.add_vertex(vc)
    

    for i in range(4):
        odometry= g2o.EdgeSE3()
        odometry.set_vertex(0,optimizer.vertex(i))
        odometry.set_vertex(1,optimizer.vertex(i+1))
        
        t=np.array([0, 0, 1])
        cam = g2o.Isometry3d(np.identity(3), t)
        odometry.set_measurement(cam)
        odometry.set_information(0.5*np.eye(6))
        optimizer.add_edge(odometry)

    
    
    
    vc = optimizer.vertex(1)
    cam = g2o.Isometry3d(vc.estimate().R, np.array([0, 0, 0.2]))
    vc.set_estimate(cam)

    vc = optimizer.vertex(2)
    cam = g2o.Isometry3d(vc.estimate().R, np.array([0, 0, 1.0]))
    vc.set_estimate(cam)


    optimizer.initialize_optimization()
    optimizer.compute_active_errors()
    print('Initial chi2 =', optimizer.chi2())

    optimizer.save('gicp.g2o')

    optimizer.set_verbose(True)
    optimizer.optimize(10)


    print('\n Third vertex should be near [0, 0, 2]')
    print('before optimization:', cam.t)
    print('after  optimization:', optimizer.vertex(4).estimate().t)
    

if __name__ == '__main__':
    main()
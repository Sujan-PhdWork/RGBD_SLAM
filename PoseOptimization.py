
import numpy as np
import g2o

def main():
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    optimizer.set_verbose(False)

    Pose=np.eye(4)
    cam = g2o.Isometry3d(Pose[:3,:3], Pose[:3,3])
    vc=g2o.VertexSE3()
    vc.set_id(0)
    vc.set_fixed(False)
    vc.set_estimate(cam)
    optimizer.add_vertex(vc)

    points=np.zeros((5,3))

    N=points.shape[0]
    for i in range(N):
        vp=g2o.VertexSBAPointXYZ()
        vp.set_estimate(points[i])
        vp.set_id(i+1)
        vp.set_fixed(True)
        optimizer.add_vertex(vp)

        z = np.random.random(2) * [640, 480]

        e=g2o.EdgeSE3ProjectXYZ()
        e.set_vertex(0,optimizer.vertex(i+1))
        e.set_vertex(1,optimizer.vertex(0))
        e.set_measurement(z)
        e.set_information(np.identity(2))
        e.set_robust_kernel(g2o.RobustKernelHuber())
        optimizer.add_edge(e)
    optimizer.initialize_optimization()
    optimizer.optimize(10)

    for i in inliers:
        vp = optimizer.vertex(i)
        error = vp.estimate() - true_points[inliers[i]]
        sse[1] += np.sum(error**2)

    

if __name__ == '__main__':
    main()
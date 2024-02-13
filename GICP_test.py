import numpy as np
import g2o
import random
from scipy.spatial import KDTree
import pcl.pcl_visualization
import pcl

# visual = pcl.pcl_visualization.CloudViewing()


def GICP(cloud_C,cloud_P):

    #cloud1: current frame cloud data
    #cloud2: previous frame cloud data


    cloud1=cloud_C.copy()
    cloud2=cloud_P.copy()
    pose=np.eye(4)


    for i in range(4):

        idx=np.random.choice(cloud2.shape[0], 500, replace=False)
        # idx=np.random.randint(size=500) 

        sampled_cloud2 = cloud2[idx,:]

        kdt=KDTree(cloud1)
        dist, indices = kdt.query(sampled_cloud2,k=1)



        # for i in range(len(cloud1)):
        #     print(f"Nearest neighbor of point {i} in matrix2 is point {indices[i]} in matrix1 with distance {dist[i]}")


        sampled_cloud1 = cloud1[indices,:]

        # print(sampled_cloud2.shape,len(indices))

        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        # robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
        opt.set_algorithm(algorithm)


        for i in range(2):
            # pose=np.dot(np.linalg.inv(f1.pose),f.pose)
            pose=np.eye(4)
            t = pose[:3,3]
            R= pose[:3,:3]

            pcam = g2o.Isometry3d(R,t)
            
            vc = g2o.VertexSE3()
            vc.set_id(i)
            vc.set_estimate(pcam)
            vc.set_fixed(i==0)
            
            opt.add_vertex(vc)

        for i in range(sampled_cloud1.shape[0]):


            meas = g2o.EdgeGICP()
            meas.pos0 = sampled_cloud2[i,:]
            meas.pos1 = sampled_cloud1[i,:]

            edge = g2o.Edge_V_V_GICP()
            edge.set_vertex(0, opt.vertex(0))
            edge.set_vertex(1, opt.vertex(1))
            edge.set_measurement(meas)
            edge.set_information(meas.prec0(1.2))
            # edge.set_robust_kernel(robust_kernel)
            opt.add_edge(edge)


        opt.initialize_optimization()
        opt.compute_active_errors()
        # if verbose:
        # print('GICP Initial chi2 =', opt.chi2())


        opt.optimize(3)

        R=opt.vertex(1).estimate().R
        t=opt.vertex(1).estimate().t

        T=np.eye(4)
        T[:3,:3]=R
        T[:3,3]=t
        pose=np.dot(T,pose)
        cloud1=np.dot(R,cloud1.T)+t.reshape(3,1)

        cloud1=cloud1.T
        # squared_errors = np.sum((cloud1 - cloud2)**2, axis=1)
        # mse = np.mean(squared_errors)
        # rmse = np.sqrt(mse)
        # print(rmse)

        
        # pltcloud1 = pcl.PointCloud()
        # pltcloud1.from_array(cloud1.astype(np.float32))
        # pltcloud2 = pcl.PointCloud()
        # pltcloud2.from_array(cloud2.astype(np.float32))

        # visual.ShowMonochromeCloud(pltcloud1)
        # visual.ShowMonochromeCloud(pltcloud2)

        # error_tree=KDTree(cloud2)
        # distances, _ = error_tree.query(cloud1, k=1)


        # # print(t)
        # mse = np.mean(distances**2)
        # rmse = np.sqrt(mse)
        # # if rmse <
        # print(rmse)
    return pose
        








        


        




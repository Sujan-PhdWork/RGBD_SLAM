import g2o
import numpy as np


R=np.eye(3)
t=np.array([1,0,0])


Pose=g2o.Isometry3d(R,t)

print(Pose.R)

ve=g2o.VertexSE3()
ve.set_estimate(Pose)

odometry=g2o.EdgeSE3()






# print(pose.R)

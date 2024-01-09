from utils import *
from scipy.spatial.transform import Rotation as R
import numpy as np





import OpenGL.GL as gl
import pangolin



def quat2T(state):
    T=np.eye(4)
    txyz=state[:3]
    q=state[3:]
    r = R.from_quat(q)
    T[:3,:3]=r.as_matrix()
    T[:3,3]=txyz
    return T





if __name__ == "__main__":



    
    dataset_path='../dataset/rgbd_dataset_freiburg1_xyz/'

    depth_paths=dataset_path+'groundtruth.txt'
    dlist=data_trajectory(depth_paths)



    
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)


    T_list=[]
    p_list=[]
    T=np.eye(4)

    for d in dlist:
        state=np.array(d)
        T=np.dot(quat2T(state),T)
        T_list.append(T)
        p_list.append(T[:3,3])
        T_array=np.array(T_list)
        p_array=np.array(p_list)


    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
        # Render OpenGL Cube
        gl.glPointSize(10)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawCameras(T_array)    

        pangolin.FinishFrame()


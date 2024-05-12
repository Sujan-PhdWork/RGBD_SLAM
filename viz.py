import pangolin
import OpenGL.GL as gl
import numpy as np


class Disp_map(object):

    def __init__(self):
        
        self.state=None

    def viewer_thread(self,q):
        self.viewer_init(640,480)
        while not pangolin.ShouldQuit():
            self.viewer_refresh(q)
    def viewer_init(self,W,H):
        pangolin.CreateWindowAndBind('Main', W, H)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(W, H, 420, 420, W//2, H//2, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0,-1,0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self,q):

        if self.state is None or not q.empty():
            self.state=q.get()
        

        ppts=np.array([d[:3,3] for d in self.state[0]])
        spts=np.array([d[:3,3] for d in self.state[1]])
        # spts=np.array(self.state[0])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)
        


        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(ppts)

        gl.glPointSize(10)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawPoints(spts)

        # print(spts.shape)
        point1 = np.array([0, 0, 0])
        point2 = np.array([1, 1, 1])
        
        # gl.glLineWidth(1)
        # gl.glColor3f(0.0, 0.0, 0.0)
        # pangolin.DrawLine(trajectory)
        
        pangolin.FinishFrame()
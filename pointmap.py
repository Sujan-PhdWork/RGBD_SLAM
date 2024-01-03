import numpy as np
import pangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue
import g2o

class Map(object):
    def __init__(self):
        self.frames=[]
        self.points=[]
        self.state=None
        self.q=None
        
        
    
   #optimizer
    def optimize(self):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)


        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        for f in self.frames:
            
            cam = g2o.Isometry3d(f.pose[:3,:3], f.pose[:3,3])
            

            vc = g2o.VertexSE3()
            vc.set_id(f.id)
            vc.set_estimate(cam)
            vc.set_fixed(f.id==0)
            optimizer.add_vertex(vc)   

        for p in self.points:

            f1,f2=p.frames
            
            #inverse transformation
            trans0 = optimizer.vertex(f1.id).estimate().inverse()
            trans1 = optimizer.vertex(f2.id).estimate().inverse()

            
            pt0 = trans0 * p.pt
            pt1 = trans1 * p.pt

            meas = g2o.EdgeGICP()
            meas.pos0 = pt0
            meas.pos1 = pt1

            edge = g2o.Edge_V_V_GICP()
            edge.set_vertex(0, optimizer.vertex(f1.id))
            edge.set_vertex(1, optimizer.vertex(f2.id))
            edge.set_measurement(meas)
            edge.set_information(meas.prec0(0.01))

            optimizer.add_edge(edge)

        
        optimizer.initialize_optimization()
        optimizer.compute_active_errors()
        print('Initial chi2 =', optimizer.chi2())

        #optimizer.save('gicp.g2o')

        optimizer.set_verbose(True)
        optimizer.optimize(10)




    
    def create_viewer(self):
        
        self.q=Queue()
    
        self.p=Process(target=self.viewer_thread,args=(self.q,))
        self.p.demon=True
        self.p.start()
    
    def viewer_thread(self,q):
        self.viewer_init(640,480)
        while not pangolin.ShouldQuit():
            self.viewer_refresh(q)


    def viewer_init(self,W,H):
        pangolin.CreateWindowAndBind('Main', W, H)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(W, H, 420, 420, W//2, H//2, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -2, 0, 0, -1, pangolin.AxisDirection.AxisX))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)


    def viewer_refresh(self,q):

        if self.state is None or not q.empty():
            self.state=q.get()
        
        # ppts=np.array([d[:3,3] for d in self.state[0]])
        spts=np.array(self.state[1])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)
        


        gl.glPointSize(10)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawCameras(self.state[0])

        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        # spts=np.array(self.state[1])

        # print("hello",spts.shape)
        pangolin.DrawPoints(spts)
        
        pangolin.FinishFrame()




    def display(self):
        poses,pts=[],[]
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((np.array(poses),np.array(pts)))

class Point(object):
    # apoint is a 3D point in the world
    #each Point is observed in multiple frame
    
    def __init__(self,mapp,loc):
        self.pt=loc #state vector
        self.frames=[]
        self.idxs=[]

        self.id=len(mapp.points)
        mapp.points.append(self)

    def add_observation(self,frame,idx):
        frame.pts[idx]=self
        self.frames.append(frame)
        self.idxs.append(idx)
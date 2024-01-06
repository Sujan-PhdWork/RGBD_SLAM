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

    

    #optimizer
    def optimize(self):
        opt=g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)


        robust_kernel = g2o.RobustKernelHuber(np.sqrt(0.991))

       
        for f in self.frames:
            
            pose=f.pose
            sbacam=g2o.SBACam(g2o.SE3Quat(pose[0:3,0:3],pose[0:3,3]))
            sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[0][2], f.K[1][2], 1.0)
            # sbacam.set_cam(1.0, 1.0, 0.0, 0.0, 1.0)

            v_se3=g2o.VertexCam()
            v_se3.set_id(f.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(f.id==0)
            opt.add_vertex(v_se3)

        PT_ID_OFFSET=0x10000
        for p in self.points:

            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(p.id+PT_ID_OFFSET)
            pt.set_estimate(p.pt[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)

            for f in p.frames:
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                if p  not in f.pts:
                    continue

                #This needs to be check
                uv=f.kps[f.pts.index(p)][:2]
                # print(uv)
                edge.set_measurement(uv)   # projection
                edge.set_information(np.eye(2))
                # edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)
            


        
        opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(20)

        #optimizer.save('gicp.g2o')

        for f in self.frames:
            est = opt.vertex(f.id).estimate()
            R = est.rotation().matrix()
            t = est.translation()
            ret=np.eye(4)
            ret[:3,:3]=R
            ret[:3,3]=t
            f.pose = ret.copy()

        for p in self.points:
            est=opt.vertex(p.id+ PT_ID_OFFSET).estimate()
            p.pt=np.array(est)









        


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
        
        ppts=np.array(self.state[0])
        spts=np.array(self.state[1])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)
        


        gl.glPointSize(10)
        gl.glColor3f(1.0, 0.0, 0.0)
        # print(ppts.shape)
        pangolin.DrawCameras(ppts)

        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        # spts=np.array(self.state[1])

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
    def __init__(self,mapp,loc):
        self.pt=loc
        self.frames=[]
        self.idxs=[]

        self.id=len(mapp.points)
        mapp.points.append(self)

    def add_observation(self,frame,idx):
        frame.pts[idx]=self
        self.frames.append(frame)
        self.idxs.append(idx)
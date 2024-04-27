import cv2
import numpy as np
from transformation import *
from ransac import *
from utils import normalize
import joblib
import pcl

# model_filename = 'BoW/kmeans_model.joblib'
# kmeans_loaded = joblib.load(model_filename)


IRt=np.eye(4)

def extractRt(model):
    R=model.params['R']
    t=model.params['t']
    c=model.params['c']

    pose=np.eye(4)
    u,s,vh = np.linalg.svd(c*R)
    pose[:3,:3]=u @ vh
    pose[:3,3]=t.reshape(3)

    return pose

def to_3D(depth,K):
  
    fx=K[0,0]
    cx=K[0,2]

    fy=K[1,1]
    cy=K[1,2]


    H=depth.shape[0]
    W=depth.shape[1]

    u=np.arange(W)
    v=np.arange(H)

    u, v = np.meshgrid(u, v)

    
    z=depth/5000.0
    x=(u-cx)*z/fx
    y=(v-cy)*z/fy

    x=np.ravel(x).reshape(-1,1)
    y=np.ravel(y).reshape(-1,1)
    z=np.ravel(z).reshape(-1,1)

    xyz=np.concatenate((x,y,z),axis=1)
    xyz = xyz[~np.all(xyz == 0, axis=1)]

    xyz=xyz.astype(np.float32)
    return xyz


def extract(img,depth,label_img):
    
    orb=cv2.ORB_create()
    feats=cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
    # feats=cv2.goodFeaturesToTrack(img.astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)

    # print(feats)
    kps=[cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in feats]
    
    # _size in numpy<1.75
    
    kps,des=orb.compute(img,kps)

    
    
    # print(des.shape)
    modified_kps=[]
    modified_des=[]
    modified_label=[]        
    
    for i,kp in enumerate(kps):
        u,v=map(lambda x: int(x),kp.pt)
        z=depth[v,u]
        l=label_img[v,u]
        if z!=0:
            modified_kps.append(kp)
            modified_des.append(des[i])
            modified_label.append(l)   
    
    modified_des=np.array(modified_des)


    

    return np.array([(kp.pt[0],kp.pt[1],
                      depth[int(round(kp.pt[1])),int(round(kp.pt[0]))]) for kp in modified_kps]),modified_des,np.array([modified_label])


def triangulate(pose1,pose2,pts1,pts2):
     ret=np.zeros((pts1.shape[0],4))
     pose1=np.linalg.inv(pose1)
    #  pose2=np.linalg.inv(pose2)
     for i, p in enumerate(zip(pts1,pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _,_,vt=np.linalg.svd(A)
        ret[i]=vt[3]
         
     return ret.T


def frames_triangulation(f1, f2, idx1, idx2, pose):
    # Extract the keypoints and descriptors for the corresponding indices
    # print(f1.kps[idx1].shape)
    
    kps1 = f1.kps[idx1]
    kps2 = f2.kps[idx2]

    kps1 /= kps1[:, 2].reshape(-1, 1)
    kps2 /= kps2[:, 2].reshape(-1, 1)

    kps1 = kps1[:, :2]
    kps2 = kps2[:, :2]


    P1=np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    # Triangulate the points

    pts4d=triangulate(np.eye(4),pose,kps2,kps1)
    pts4d /= pts4d[3, :]

    
    return pts4d 



def match_by_segmentation(f1,f2):
    
    
    # f1 is current frame f2 is previous frame
    ## we will first check how many labels are in last frame
    ## we will use all the points to calculate the transformation
    ## we will create a function that will take the transformation and return the 3d points
        ## ->(will check later) We will reject bad triangulation 
    # we will then check the matched point triangulation depth and original depth
    # for covariance measurement we will for now taking maximum condition number of the covariance matrix
    # if the covariance is less than threshold we will consider it as a good match
    # then we will take all the labels except an random one
    # and check if covariance is increase or decrease if its increase then we will reject all the point that are in that label
    # if its decreases we will accept all the points that are in that label and make that segment as good segment


    
    frame_lable=np.unique(f2.label)
    
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.knnMatch(f1.des,f2.des,k=2)
    ret=[]
    idx1,idx2=[],[]

    pose=None
    for m,n in matches:
        if m.distance <0.75*n.distance:
            if m.distance < 30:
                

                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                
                kp1=f1.kps[m.queryIdx]
                kp2=f2.kps[m.trainIdx]

                # kp1[idx]-> the keypoint in previous frame with id =idx            
                ret.append((kp1,kp2))
    
    ret=np.array(ret).astype(np.float32)
    idx1=np.array(idx1)
    idx2=np.array(idx2)
    
    ransac=RANSAC(ret,Transformation(),8,0.05,500)
    model,inliers,error=ransac.solve()
    ret=ret[inliers]
    idx1=idx1[inliers]
    idx2=idx2[inliers]
    

    pose=extractRt(model)



    # Points on the previous frame

    point_p=frames_triangulation(f1, f2, idx1, idx2, pose)


    # good_pts4d=np.abs(point_p[3,:]>0.005) & (point_p[2,:]>0)

    # print(point_p.shape,good_pts4d.shape)

    # point_p=point_p[:,good_pts4d]
    # ret=ret[good_pts4d]


    diff=ret[:,1,:]-point_p[:3,:].T

    mean=np.mean(diff,axis=0)

    # np.outer(kp2[:,i]-muy.T,kp1[:,i]-mux.T)
    cov=np.cov((diff).T)
    print(cov)

    # eg= np.linalg.eigvals(cov)
    cond= np.linalg.cond(cov)
    # det=np.linalg.det(cov)
    print(cond)


    if cond>0.7:
        return idx1,idx2,pose
    

    
    return point_p
    # print(pts4d)



def match(f1,f2):
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.knnMatch(f1.des,f2.des,k=2)
    
    #low's ratio test
    ret=[]
    idx1,idx2=[],[]
    pose=None

    for m,n in matches:
        if m.distance <0.75*n.distance:
            if m.distance < 30:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

                # idx1-> id of the keypoint in previous frame
                # idx2-> id of the keypoint in current frame

                kp1=f1.kps[m.queryIdx]
                kp2=f2.kps[m.trainIdx]

                # kp1[idx]-> the keypoint in previous frame with id =idx            
                ret.append((kp1,kp2))

    assert len(ret)>=3
    ret=np.array(ret).astype(np.float32)


    idx1=np.array(idx1)
    idx2=np.array(idx2)

    # ret[:,0,:2]=normalize(ret[:,0,:2],f1.Kinv)
    # ret[:,1,:2]=normalize(ret[:,1,:2],f2.Kinv)
    
    ransac=RANSAC(ret,Transformation(),3,0.05,100)
    model,inliers,error=ransac.solve()
    # print(error)
    idx1=idx1[inliers]
    idx2=idx2[inliers]

    pose=extractRt(model)

    return idx1,idx2,pose


def GICP(cloud2,cloud1):
    icp = cloud1.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(cloud1, cloud2)
    # print('has converged:' + str(converged) + ' score: ' + str(fitness))
    # print(str(transf))
    return transf
    






class Frame(object):
    def __init__(self,mapp,img,depth,K):
        
        
        self.cloud=to_3D(depth,K)

        

        depth_Z= depth.reshape((-1,1))    
        depth_Z = np.float32(depth_Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.sample = 10 # this need to be tune
        ret,labels,centers=cv2.kmeans(depth_Z,self.sample,None,criteria,50,cv2.KMEANS_RANDOM_CENTERS)
        # self.label_img=labels.reshape(depth.shape)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        label_img=segmented_data.reshape(depth.shape)
        segment_colors = np.random.randint(0, 256, (self.sample, 3), dtype=np.uint8)
        colored_segmented_data = segment_colors[labels.flatten()]
        self.colored_segmented_img = colored_segmented_data.reshape(depth.shape[0],depth.shape[1], 3)

        self.pts,self.des,self.label=extract(img,depth,label_img)


    
        # center = np.uint8(center)
        # print(label.shape, depth_Z.shape,depth.shape)
        # res = center[label.flatten()]
        # res2 = res.reshape((depth.shape))
        # cv2.imshow('res2',res2) 


        self.Ihist=None
        #pts is 3d points unlormalize point

        self.K=K

        
        self.Kinv=np.linalg.inv(K)

        self.kps=self.pts.copy()
        # self.pts[:,:,2]=normalize(self.kps[:,:2],self.Kinv)

        self.kps[:,:2]=normalize(self.kps[:,:2],self.Kinv)
        
        #kps is 3d  normalize point 
        self.kps[:,2]=self.kps[:,2]/5000.0 #factor 
        
        self.kps[:,0]=self.kps[:,0]*self.kps[:,2]
        self.kps[:,1]=self.kps[:,1]*self.kps[:,2]

        self.pose=IRt
        self.Rpose=IRt
        self.isKey=False
        self.id=len(mapp.frames)
        







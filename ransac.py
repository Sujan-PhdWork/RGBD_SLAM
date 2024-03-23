import numpy as np 


class RANSAC:
    def __init__(self,f1,f2,matches21,model,n,th,iter):
        
        self.f1=f1
        self.f2=f2
        self.matches=matches21

        # coordinate_array=np.array(matches)
        # self.kp1=coordinate_array[:,0,:].T
        # self.kp2=coordinate_array[:,1,:].T
        
        self.sample=n
        self.threshold=th
        self.iter=iter
        self.model=model
        
    
    def solve(self):
        i=0
        best_inlier=0
        best_params={}
        # T=(1-self.eta)*self.pt1.shape[1]
        # print(T)
        while(i<self.iter):

            #select random sample
            # print(self.pt1.shape[1])
            
            


            idxs=np.random.randint(0,self.matches.shape[0],size=self.sample) 
            

            match1=self.matches[idxs,1]
            match2=self.matches[idxs,0]


            pts1=self.f1.kps[match1,:]
            pts2=self.f2.kps[match2,:]
            

            #compute model parameter
            
            self.model.compute_parameter(pts1.T,pts2.T)
            
            #calculate the reprojection error
            error=self.model.error(self.f1.kps[self.matches[:,1]].T,self.f2.kps[self.matches[:,0]].T)

            if error is None:
                continue
            
            mask=error<self.threshold 
            
                    
            if (np.sum(mask)>np.sum(best_inlier)):
                # print(np.sum(mask))  
                best_params=self.model.params
                best_inlier=mask
            # if (np.sum(mask)>T):
            #     break
        
                

            # print(inliers_pts1.size)
            i+=1
        self.model.params=best_params
        
        return self.model,best_inlier,np.sum(error)
        # plt.show()

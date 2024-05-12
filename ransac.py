import numpy as np 


class RANSAC:
    def __init__(self,matches,model,n,th,iter):
        coordinate_array=np.array(matches)
        self.kp1=coordinate_array[:,0,:].T
        self.kp2=coordinate_array[:,1,:].T
        
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

            idxs=np.random.randint(0,self.kp1.shape[1],size=self.sample) 
                  
            
            pts1=self.kp1[:,idxs]
            pts2=self.kp2[:,idxs]

            #compute model parameter
            
            self.model.compute_parameter(pts1,pts2)
            
            #calculate the reprojection error
            error=self.model.error(self.kp1,self.kp2)

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

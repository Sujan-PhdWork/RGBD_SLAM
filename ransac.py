class RANSAC:
    def __init__(self,matches,model,n,th,eta=0.3):
        self.pt1=pt1 #need correction
        self.pt2=pt2 
        self.sample=n
        self.threshold=th
        self.eta=eta
        self.model=model
    
    def solve(self):
        i=0
        best_inlier=0
        T=(1-self.eta)*self.pt1.shape[1]
        # print(T)
        while(i<100):

            #select random sample
            # print(self.pt1.shape[1])

            idxs=np.random.randint(0,self.pt1.shape[0],size=self.sample)        
            pts1=self.pt1[idxs,:]
            pts2=self.pt2[idxs,:]

            #compute model parameter
            
            self.model.compute_parameter(pts1,pts2)
            
            #calculate the reprojection error
            error=self.model.error2(self.pt1,self.pt2)
            
            if error is None:
                continue
            
            mask=error<self.threshold 
            
                    
            if (np.sum(mask)>np.sum(best_inlier)):
                print(np.sum(mask))  
                self.model.params=params
                best_inlier=mask
            # if (np.sum(mask)>T):
            #     break
        
                

            # print(inliers_pts1.size)
            i+=1
        
        return self.model,best_inlier,np.sum(error)
        # plt.show()

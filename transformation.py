import numpy as np

def transformation(kp1,kp2):
    
    
    mux=np.mean(kp1,axis=1).reshape(-1,1)
    muy=np.mean(kp2,axis=1).reshape(-1,1)

    var_x=np.mean(np.square(np.linalg.norm(kp1-mux,axis=0)))
    var_y=np.mean(np.square(np.linalg.norm(kp2-muy,axis=0)))




    outer_products=[]

    for i in range(kp1.shape[1]):
        outer_products.append(np.outer(kp2[:,i]-muy.T,kp1[:,i]-mux.T))

    

    outer_products=np.array(outer_products)

    # print(var_x_products)
    
    cov=np.mean(outer_products,axis=0)
    


    U,d,VT=np.linalg.svd(cov)

    D=np.diag(d)
    
    S=np.eye(D.shape[0])
    

    # print(np.linalg.det(U)*np.linalg.det(VT))
    if (np.linalg.det(U)*np.linalg.det(VT))<0:
        S[-1,-1]=-1
    # if np.linalg.det(cov)<1:
    #     

    R=np.dot(U,np.dot(S,VT))
    c=(1/var_x)*(np.trace(D*S))
    t=muy-c*np.dot(R,mux)
    
    error=var_y-((np.trace(D*S))**2)/var_x

    params={"R": R,"t":t,"c":c}
    return (params,error)




class Transformation:
    def __init__(self):
        self.params=None
        
    def compute_parameter(self,pt1,pt2):
        self.params,_=transformation(pt1,pt2)

    def error (self,kp1,kp2):
        
        R=self.params['R']
        t=self.params['t']
        c=self.params['c']

        estimated_kp2=c*np.dot(R,kp1)+t
        error=np.linalg.norm(estimated_kp2-kp2,axis=0)
        return(error)
        
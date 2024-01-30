import numpy as np
from frame import match
import random
from threading import Thread,Lock
from pointmap import EDGE


def lc_process(mapp,n):
    t1=Thread(target=loop_closure,args=(mapp,n))
    t1.start()


def TF_IDF(frames):
    
    N=len(frames)
    IDF=np.zeros_like(frames[-1].hist)+1e-0

    for f in frames:
        hist=f.hist
        # All the histogram which is greater than zero
        H_1=hist>0
        # It will add all those index to see who has that hist
        IDF=IDF+H_1.astype(int)
    
    for f in frames:
        hist=f.hist
        normalize=hist/np.sum(hist)
        
        Ihist=normalize*np.log(N/IDF)
        f.Ihist=Ihist
    

def cos_distance(hist1,hist2):
    
    hist1_mag=np.sqrt(np.dot(hist1.T,hist1))
    hist2_mag=np.sqrt(np.dot(hist2.T,hist2))
    cossim=np.dot(hist1.T,hist2)/(hist1_mag*hist2_mag)
    dcos=1-cossim
    # print(dcos.shape)

    return (dcos)




def loop_closure(mapp,n=20):

    sampled_frames=random.sample(mapp.frames[:-3], n-3)
    
    sampled_frames.append(mapp.frames[-1])
    sampled_frames.append(mapp.frames[-2])
    sampled_frames.append(mapp.frames[-3])
    
    TF_IDF(sampled_frames)

    f1=sampled_frames[-1]
    f2=sampled_frames[-2]
    f3=sampled_frames[-3]


    dcos_list=[]
    for f in sampled_frames[:-3]:

        dcos1=cos_distance(f.Ihist,f1.Ihist)
        # dcos2=cos_distance(f.Ihist,f2.Ihist)
        # dcos3=cos_distance(f.Ihist,f3.Ihist)
        if dcos1 <0.3:
            print(dcos1)
            # print(dcos1,dcos2,dcos3)
            _,_,pose=match(f,f1)
            EDGE(mapp,f.id,f1.id,pose)
        dcos_list.append(dcos1)
    # print(min(dcos_list))





    
    


    
  
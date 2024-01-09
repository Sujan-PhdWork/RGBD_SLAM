import numpy as np
from frame import match
import random
from threading import Thread,Lock
from pointmap import EDGE


def lc_process(mapp,n):
    t1=Thread(target=loop_closure,args=(mapp,n))
    t1.start()

def loop_closure(mapp,n=20):

    
    sampled_frames=random.sample(mapp.frames[:-1], n-1)
    sampled_frames.append(mapp.frames[-1])

    f1=sampled_frames[-1]
    idx1s=[]
    idx2s=[]
    poses=[]
    
    for f in sampled_frames[:-1]:
        
        # print(f1.id,f.id)
        # id1=f1.id
        # id2=f.id
        idx1,idx2,pose=match(f,f1)
        poses.append(pose)
        idx1s.append(len(idx1))
        idx2s.append(len(idx2))

    idx1s=np.array(idx1s)
    
    max_index=np.argmax(idx1s)

    f_final=sampled_frames[max_index]
    
    f_pose=poses[max_index]
    
    EDGE(mapp,f_final.id,f.id,f_pose)
    # EDGE(mapp,)
    


    
  
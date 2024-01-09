import numpy as np
from frame import match
import random
from multiprocessing import Process,Lock


def loop_closure(mapp,n=20):

    
    sampled_frames=random.sample(mapp.frames[:-1], n-1)
    sampled_frames.append(mapp.frames[-1])

    for f in sampled_frames[:-1]:
        f1=sampled_frames[-1]
        idx1,idx2,pose=match(f1,f)
        print(len(idx1))
  
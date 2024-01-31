from frame import match
import numpy as np


def local_mapping(map):
    for f in map.frames:
        if f.id==f.keyid:
            f_p=f
            print("This is a key id",f.id)
            continue
        else:
            idx1,idx2,pose=match(f,f_p)
            f.pose=np.dot(pose,f.pose)
            






    # local_frames=[f.id for f in map.frames]
    # print("new local frame", local_frames)
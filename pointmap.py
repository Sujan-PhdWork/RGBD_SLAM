import numpy as np
import pangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue



class Map(object):
    def __init__(self):
        self.frames=[]
        self.points=[]


    def display(self):
        for f in self.frames:
            print(f.id)
            print(f.pose)

class Point(object):
    def __init__(self,mapp):
        self.frames=[]
        self.points=[]
        mapp.points.append(self)

    def add_observation(self,frame,pts):
        self.frames.append(frame)
        self.points.append(pts)
import cv2
import numpy as np


def disp(img,WindowName):
    cv2.imshow(WindowName,img)


def add_ones(x):
    return np.concatenate([x,np.ones((x.shape[0],1))],axis=1)


def normalize(pts,Kinv):
        return np.dot(Kinv,add_ones(pts).T).T[:,0:2]

def denormalize(pt,K):
        K_copy=K.copy()
        K_copy[:2,:2]=K[:2,:2]/(pt[2]+1e-09)
        ret=np.dot(K_copy,np.array([pt[0],pt[1],1.0]))
        return int(round(ret[0])),int(round(ret[1])),pt[2]

def denormalize2(pt,K):
        K_copy=K.copy()
        # K_copy[:2,:2]=K[:2,:2]/(pt[2]+1e-09)
        ret=np.dot(K_copy,np.array([pt[0],pt[1],1.0]))
        return int(ret[0]),int(ret[1])



def data(file_path):

    file_list=[]

    # Specify the file path
    # file_path = '../rgbd_dataset_freiburg1_xyz/depth.txt'

    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:

            for line in file:

                if not line.startswith('#'):
                    words=line.split()
                    file_list.append(words[1])
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    except Exception as e:
        print(f"An error occurred: {e}")

    
    return (file_list)




def data_trajectory(file_path):

    state_list=[]

    # Specify the file path
    # file_path = '../rgbd_dataset_freiburg1_xyz/depth.txt'

    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:

            for line in file:

                if not line.startswith('#'):
                    words=line.split()
                    state=[float(s) for s in words[1:]]
                    state_list.append(state)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    except Exception as e:
        print(f"An error occurred: {e}")

    
    return (state_list)


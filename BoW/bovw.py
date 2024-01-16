import sys
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
import joblib

sys.path.append('../')

from utils import data





def finding_Kmean(des_array):
    global kmeans
    kmeans.fit(des_array,)
    # print(kmeans.labels_)

def process_img(img):
    orb=cv2.ORB_create(100)
    feats=cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000,qualityLevel=0.01,minDistance=3)
    kps=[cv2.KeyPoint(x=f[0][0],y=f[0][1],size=20) for f in feats]
    kps,des=orb.compute(img,kps)
    return des


def descriptor_list(image_list,dataset_path):

      for i in range(len(image_list)-1):
            frame=cv2.imread(dataset_path+image_list[i])
            if np.shape(frame) == ():
                  continue
            des=process_img(frame)
            if i==0:
                  des_array=des
            else:
                  des_array=np.vstack((des_array,des))
      return(des_array)

def update_kmean(dataset_path):
    global kmeans
    
    rgb_paths=dataset_path+'rgb.txt'
    
    image_list=data(rgb_paths)
    if image_list is None:
        return
    
    des_array=descriptor_list(image_list,dataset_path)
    
    finding_Kmean(des_array)
        
if __name__ == "__main__":
    
    
    # #initializa praser
    parser=argparse.ArgumentParser()
    
    parser.add_argument("-f", "--Filename" ,nargs='*',type=str, help = "Path of the dataset")
    parser.add_argument("-n", "--Cluster",type=int,help = "Number of cluster")   
    
    args = parser.parse_args()
    # 
    N_cluster=int(args.Cluster)
    kmeans = KMeans(n_clusters=N_cluster, n_init='auto',random_state=10,)
    
    for f in args.Filename:
        print("Displaying Filename as: % s" % f)
        update_kmean(f)
    
    dataset_path1='../../dataset/rgbd_dataset_freiburg1_floor/'
    print(dataset_path1)
    
    update_kmean(dataset_path1)
    dataset_path2='../../dataset/rgbd_dataset_freiburg2_xyz/'
    print(dataset_path2)

    update_kmean(dataset_path2)
    # dataset_path3='../../dataset/rgbd_dataset_freiburg2_floor/'
    # update_kmean(dataset_path3)
    dataset_path4='../../dataset/rgbd_dataset_freiburg2_rpy/'
    print(dataset_path4)

    update_kmean(dataset_path4)



    
    while True:
        key=input("Enter 's' or 'n': ")
        if key == 's':
            print(key)
            break
        if key=='n':
            Filename=input("Enter File path: ")
            print(type(Filename))
            print("Displaying Filename as: % s" % Filename)
            update_kmean(Filename)

    # print(kmeans.labels_)
    model_filename = 'kmeans_model.joblib'
    joblib.dump(kmeans, model_filename)

    print(f"KMeans model saved to {model_filename}")
    
         
         
    

    


import cv2
import numpy as np
from utils import *
import matplotlib.pyplot as plt





def process_img(img,depth,mod_depth):

    depth_Z=depth.flatten()
    depth_Z=depth_Z[~(depth_Z==0)]
    hist, bins = np.histogram(depth_Z, bins=10000, range=[0,10000])

    # Plot the histogram
    plt.plot(hist, color='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Grayscale Frame')
    plt.xlim([0, 10000])

    # Display the plot
    plt.pause(0.01)
    plt.clf()
    # plt.clear()
    # depth_Z= depth.reshape((-1,1))
    # depth_Z=depth_Z[~(depth_Z==0)]
    # plt.hist(depth_Z, bins=5000, color='blue', alpha=0.7)
    # plt.pause(0.05)

    

    # # moddepth = np.mean(depth,axis=2)
    # depth_Z= depth.reshape((-1,1))    
    # depth_Z = np.float32(depth_Z)
    # criteria = (cv2.TERM_CRITits two neighborsERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # sample = 10 # this need to be tune
    # ret,labels,centers=cv2.kmeans(depth_Z, sample,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # centers = np.uint8(centers)
    # segmented_data = centers[labels.flatten()]
    # label_img=segmented_data.reshape(depth.shape)

    # segment_colors = np.random.randint(0, 256, (sample, 3), dtype=np.uint8)

    # colored_segmented_data = segment_colors[labels.flatten()]
    # colored_segmented_img = colored_segmented_data.reshape(depth.shape[0],depth.shape[1], 3)



    # cv2.imshow("Kmean",colored_segmented_img)
    disp(img,"RGB")
    disp(depth,"Depth")

def optimize_frame(mapp):
    mapp.optimize()
    


if __name__ == "__main__":
    dataset_path='../dataset/rgbd_dataset_freiburg3_walking_xyz_validation/'
    # dataset_path='../dataset/rgbd_dataset_freiburg1_floor/'
    depth_paths=dataset_path+'depth.txt'
    dlist=data(depth_paths)

    rgb_paths=dataset_path+'rgb.txt'
    ilist=data(rgb_paths)

    # Loop.lc.event.set()

    plt.show()
    for i in range(len(dlist)):

        frame=cv2.imread(dataset_path+ilist[i]) # 8 bit image
        depth=cv2.imread(dataset_path+dlist[i],-1) # 16 bit monochorme image
        mod_depth=cv2.imread(dataset_path+dlist[i],0)
        
        
        process_img(frame,depth,mod_depth)


        if frame is None:
            print("End of frame")
            break
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # optimize_frame(mapp)
    # mapp.display()


        




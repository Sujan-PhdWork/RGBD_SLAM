
import cv2
import numpy as np
from utils import *
import matplotlib.pyplot as plt

datas=[]


def non_empty_values(data):
    global datas
    if isinstance(data, tuple):
        for item in data:
            non_empty_values(item)
    elif data is not None:
       datas.append(data) 




def maximum_hist(hist,bins):
   
    hist=np.array(hist)
    bins=np.array(bins)
    # print("hello",hist)
    # print("gelo",bins)
    max_hist=[]
    max_bins=[]
    for i  in range(1,hist.shape[0]-1):
        y_prev=hist[i-1]
        y_next=hist[i+1]
        if hist[i]>y_next and hist[i]>y_prev:
            max_hist.append(hist[i])
            max_bins.append(bins[i])
    
    return max_hist,max_bins


def max_split(hist,bins):
    if len(bins) < 2 or abs(bins[0]-bins[-1])<200:
        if len(bins)==0:
            return
        return hist,bins 
    else:
        # print(len(bins),abs(bins[0]-bins[-1]))
        for i in range(len(bins)-1):
            
            if abs(bins[i+1]-bins[i])>200:
                breakpoint=i
                break

        #  = i
        left_half_hist = hist[:breakpoint+1]
        left_half_bins = bins[:breakpoint+1]

        left_max_hist,left_max_bins=maximum_hist(left_half_hist,left_half_bins)
        # # print(left_max_bins)
        right_half_hist = hist[breakpoint+1:]
        right_half_bins = bins[breakpoint+1:]
        right_max_hist,right_max_bins=maximum_hist(right_half_hist,right_half_bins)
        # return max_split(right_half_hist,right_half_bins)
        return max_split(right_max_hist,right_max_bins), max_split(left_max_hist,left_max_bins)





def process_img(img,depth,mod_depth):
    global datas
    depth_Z=depth.flatten()
    depth_Z=depth_Z[~(depth_Z==0)]
    hist, bins = np.histogram(depth_Z, bins=depth_Z.shape[0], range=[0,depth_Z.shape[0]])
    plt.plot(hist, color='black')
    
    max_hist,max_bins=maximum_hist(hist,bins)


    plt.scatter(max_bins,max_hist, color='red')
    T=40
    k=len(max_hist)

    results=max_split(max_hist,max_bins)

    datas=[]
    non_empty_values(results)
    print(datas)
    my_hist=[]
    my_bin=[]
    for i in range(len(datas)):
        if i%2==0:
            my_hist.append(datas[i])
        else:
            my_bin.append(datas[i])
    
    
    plt.scatter(my_bin,my_hist, color='green')






    # Plot the histogram
    
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Grayscale Frame')
    plt.xlim([0, depth_Z.shape[0]])

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


        




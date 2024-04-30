
import cv2
import numpy as np
from utils import *

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator


last_depth=None
prev=None

w=640
h=480

X= np.arange(0,w,1)
Y= np.arange(0,h,1)

X,Y= np.meshgrid(X,Y)


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})




def process_img(img,depth_org,mod_depth):
    global last_depth,X,Y,fig,ax,prev
    h,w=depth_org.shape[:2]
    # print(h,w)
    depth=depth_org.astype(np.double)
    next=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    if last_depth is None:
        last_depth=depth
        prev=next
        return
        
    diff=depth-last_depth

    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(img)
    hsv[..., 1] = 255
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # normalized_diff = cv2.normalize(diff, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    # # colored_diff = cv2.applyColorMap(normalized_diff, cv2.COLORMAP_JET)
    
    # # diff[(np.abs(diff)>1000.0)]=0.0
    
    disp(bgr,"Depth")
    
    # Z=diff
    # # R = np.sqrt(X**2 + Y**2)
    # # Z = np.sin(R)
    
   
    # # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                    linewidth=0.1, antialiased=False)
    
    # # ax.set_zlim(0, 100000)
    # # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically    
    # ax.zaxis.set_major_formatter('{x:.02f}')
    # # qfig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.pause(0.01)
    # plt.clf()





    # ret, thresh = cv2.threshold(mod_depth,0,50,cv2.THRESH_OTSU)
    # disp(thresh,"Threshold")
    disp(img,"RGB")
    disp(depth_org,"Depth2")
    last_depth=depth
    prev=next

    


if __name__ == "__main__":
    dataset_path='../dataset/rgbd_dataset_freiburg3_walking_xyz_validation/'
    # dataset_path='../dataset/rgbd_dataset_freiburg1_xyz/'
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


        




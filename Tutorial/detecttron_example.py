import torch
import detectron2
import numpy as np
import cv2
import time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.data import MetadataCatalog


cfg_pan = get_cfg()
cfg_pan.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml'))
cfg_pan.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml')

def process_img(frame):

    global cfg_pan

    width=frame.shape[1]
    height=frame.shape[0]
    # dim = (width, height)

    myNewImage=cv2.resize()
    myNewImage=frame

    # cfg_pan.MODEL.DEVICE = "cpu" # if you have Cuda, dont need this line
    predictor= DefaultPredictor(cfg_pan)

    t=time.time()
    predictions=predictor(myNewImage)
    eps=time.time()-t
    print(eps)
    instances=predictions["instances"].to("cpu")

    # print(instances.)
    # print(predictions["pred_masks"])

    mask_image=instances.pred_masks.numpy()

    mask_image=np.sum(mask_image,axis=0)
    mask_image=mask_image.astype(np.uint8)

    cv2.imshow("img", myNewImage)
    cv2.imshow("predict", mask_image*255)




vid = cv2.VideoCapture(0) 

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    
    process_img(frame)

    # Display the resulting frame 
    # cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
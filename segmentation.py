import torch
import detectron2
import numpy as np
import time
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg



def segmentation(frame: np.ndarray, prob=0.95,viz=True):
    
    cfg_pan = get_cfg()
    cfg_pan.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml'))
    cfg_pan.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml')
    
    w=frame.shape[1]
    h=frame.shape[0]
    

    predictor= DefaultPredictor(cfg_pan)

    
    
    t=time.time()
    predictions=predictor(frame)
    eps=time.time()-t
    print("Prediction time: ",eps)


    instances=predictions["instances"].to("cpu")
    
    detected_class_indexes = instances.pred_classes

    detected_scores= instances.scores

    mask_image=instances.pred_masks.numpy()
    

    total_mask=np.zeros(mask_image[0].shape,dtype=np.uint8)
    # background = np.full(frame.shape, 255, dtype=np.uint8)
    # bk = cv2.bitwise_or(background, background, mask=total_mask)

    lable_image=np.zeros([h,w],dtype=np.int8)
    

    for i in range(mask_image.shape[0]):
        if detected_scores[i]>prob:
            lable_image[mask_image[i]]=i+1
            total_mask=total_mask+mask_image[i]
    total_mask_bk=np.invert(total_mask*255)
    background=cv2.bitwise_or(frame, frame, mask=total_mask_bk)
    
    if viz:
        segment_colors = np.random.randint(0, 256, (len(detected_class_indexes)+1, 3), dtype=np.uint8)
        colored_segmented_data = segment_colors[lable_image.flatten()]
        colored_segmented_img = colored_segmented_data.reshape(h,w, 3)
        return lable_image,colored_segmented_img,background
    else:
        return lable_image,background
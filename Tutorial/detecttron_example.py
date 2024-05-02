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


imagePath = "demo.jpg"
myNewImage = cv2.imread(imagePath)
scale_precent = 30

width = int(myNewImage.shape[1] * scale_precent / 100)
height = int(myNewImage.shape[0] * scale_precent / 100)
dim = (width, height)

myNewImage = cv2.resize(myNewImage, dim, interpolation=cv2.INTER_AREA)
# Panoptic Segmentation = Instance Segmentation + Semnatic Segmentation

cfg_pan = get_cfg()
cfg_pan.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml'))
cfg_pan.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml')
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
# mask_image=mask_image
print(mask_image.shape,myNewImage.shape)



# viz = Visualizer (myNewImage[:, :,::-1], MetadataCatalog.get(cfg_pan.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)

# output= viz.draw_instance_predictions(predictions["instances"].to("cpu"))

# img = output.get_image()[:, :, ::-1]

cv2.imshow("img", myNewImage)
cv2.imshow("predict", mask_image*255)
cv2.waitKey(0)

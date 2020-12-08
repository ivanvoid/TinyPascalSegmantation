# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

from os.path import join
import torch
import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from itertools import groupby
from pycocotools import mask as maskutil

from utils import get_pascal_dict


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], 
                                          rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle


def main():
    cfg = get_cfg()
    # Get config file for model for this run
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file('configs/custum_config.yaml')
    cfg.DATASETS.TRAIN = ("pascal_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # 512  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21 
    
    
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_pascal_dict('data/pascal_test.json', 
                                    test=True)

    submission = []

    for sample in dataset_dicts:
        im = cv2.imread(sample["file_name"])
        outputs = predictor(im)  

        inst = outputs['instances']
        for i in range(len(outputs['instances'])):

            report = {
            'image_id': None,
            'score': None,
            'category_id': None,
            'segmentation': None
            }

            report['image_id'] = sample['image_id']
            report['score'] = float(inst.scores.detach().cpu().numpy()[i])
            report['category_id'] = int(inst.pred_classes.detach().cpu().numpy()[i])
            seg = inst.pred_masks.detach().cpu().numpy()[i]
            report['segmentation'] = binary_mask_to_rle(seg.astype('uint8'))

            submission.append(report)
            
    ''' Save prediction into json file '''
    with open(cfg.OUTPUT_DIR + '/submission.json', 'w') as outfile:
        json.dump(submission, outfile)



if __name__ == '__main__':
    main()
    print('Done.')

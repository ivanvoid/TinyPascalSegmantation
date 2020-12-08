# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from os.path import join
import torch
import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from detectron2.engine import DefaultTrainer

from utils import get_pascal_dict


def get_classes(train_json_path):
    with open(train_json_path) as json_file:
        train_labels_d = json.load(json_file)
    classes = [cat['name'] for cat in train_labels_d['categories']]
    
    n_cls = ['0']
    n_cls.extend(classes)
    classes = n_cls
    
    return classes


def train(classes):
    cfg = get_cfg()
    # Get config file for model for this run
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file('configs/custum_config.yaml')
    cfg.DATASETS.TRAIN = ("pascal_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    # HERE I NOT LOADING THE WEIGHTS
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 4000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # 512  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)  

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


def main():
    classes = get_classes('data/pascal_train.json')
#     print(classes)
    
    DatasetCatalog.register('pascal_train', 
                        lambda: get_pascal_dict(
                            'data/pascal_train.json'))

    MetadataCatalog.get("pascal_train").set(thing_classes=classes)
    pascal_metadata = MetadataCatalog.get("pascal_train")
    
    
    train(classes)


if __name__ == '__main__':
    main()
    print('Done.')

import argparse

import torch
import numpy as np
import json
from os.path import join
from itertools import groupby

from scripts.dataset import TinyPascal
from scripts.config import cfg  # global singleton usage pattern

from pycocotools import mask as maskutil

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import det_utils.transforms as T
import det_utils.utils as utils

from det_utils.engine import train_one_epoch, evaluate
import det_utils.utils as utils 


def pars_args():
    '''Parse stuff'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--confpath', type=str, required=True, default="./configs/experiment_1.yaml", help="Path to experiment config")
    args = parser.parse_args()
    
    return args


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_config(current_experiment_file):
    cfg.merge_from_file(current_experiment_file)
    cfg.freeze()


def get_dataloaders(args):
    dataset = TinyPascal(args['datadir']+"/pascal_train.json",  get_transform(train=True))
    dataset_test = TinyPascal(args['datadir']+"/pascal_test.json",  get_transform(train=False), train=False)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args['train_bs'], shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args['test_bs'], shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    return data_loader, data_loader_test


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_dataloaders(args):
    dataset = TinyPascal(args['datadir']+"/pascal_train.json",  get_transform(train=True))
    dataset_test = TinyPascal(args['datadir']+"/pascal_test.json",  get_transform(train=False), train=False)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args['train_bs'], shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args['test_bs'], shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    return data_loader, data_loader_test


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle


def main(args):
    print('Load experemental config...')
    current_experiment_file = args.confpath
    load_config(current_experiment_file)
    
    print('Load model...')
    n_classes = cfg.SYSTEM.N_CLASSES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = get_model_instance_segmentation(n_classes)

    model.load_state_dict(torch.load(cfg.SYSTEM.MODEL_PATH))
    model.eval()
    model.to(device)
    
    print('Load test data...')
    dl_args = {'datadir': cfg.SYSTEM.DATA_FOLDER, 
        'train_bs':cfg.TRAIN.BATCH_SIZE,
        'test_bs': cfg.TEST.BATCH_SIZE
       }
    _, test_dl = get_dataloaders(dl_args)
    
    
    print('Inference...')
    sublission = []
    for i, batch in enumerate(test_dl):
        image = batch[0][0].to(device).unsqueeze(0)

        prediction = model(image) # run inference of your model
        
        if len(prediction[0]['labels']) > 0:  # If any objects are detected in this image
            for j in range(len(prediction[0]['labels'])):
                if float(prediction[0]['scores'][j]) > 0.7: # if confidence > 80%
    #                 # save information of the instance in a dictionary then append on coco_dt list
                    pred = {}

                    b_mask = prediction[0]['masks'][0][0].cpu().detach().numpy()
                    b_mask = (b_mask > 0.2).astype('uint8')
    #                 print(b_mask.shape)

                    pred['image_id'] = test_dl.dataset.keys[i] # this imgid must be same as the key of test.json
                    pred['category_id'] = int(prediction[0]['labels'][j])
                    pred['segmentation'] = binary_mask_to_rle(b_mask)  # save binary mask to RLE, e.g. 512x512 -> rle
                    pred['score'] = float(prediction[0]['scores'][j])
                    sublission.append(pred)
#         print(sublission)
        
    ''' Save prediction into json file '''
    with open('outputs/submission.json', 'w') as outfile:
        json.dump(sublission, outfile)

    
if __name__ == '__main__':
    args = pars_args()
    main(args)
    print('Done.')
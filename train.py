import argparse

import torch
import numpy as np
import json
from os.path import join

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from scripts.dataset import TinyPascal
from scripts.config import cfg  # global singleton usage pattern

import det_utils.transforms as T
import det_utils.utils as utils

from det_utils.engine import train_one_epoch, evaluate
import det_utils.utils


def pars_args():
    '''Parse stuff'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--confpath', type=str, required=True,
                        default="./configs/ex1.yaml",
                        help="Path to experiment config")
    args = parser.parse_args()

    return args


def load_config(current_experiment_file):
    cfg.merge_from_file(current_experiment_file)
    cfg.freeze()


def get_dataloaders(args):
    dataset = TinyPascal(args['datadir']+"/pascal_train.json",
                         get_transform(train=True))
    dataset_test = TinyPascal(args['datadir']+"/pascal_test.json",
                              get_transform(train=False), train=False)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args['train_bs'], shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args['test_bs'], shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    return data_loader, data_loader_test


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model
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


def training_loop(model, train_dl, device):
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(cfg.TRAIN.N_EPOCHS):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dl, device,
                        epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

    return model


def main(args):
    print('Load experemental config...')
    current_experiment_file = args.confpath
    load_config(current_experiment_file)

    print('Load dataloaders...')
    dl_args = {'datadir': cfg.SYSTEM.DATA_FOLDER,
               'train_bs': cfg.TRAIN.BATCH_SIZE,
               'test_bs': cfg.TEST.BATCH_SIZE
               }
    train_dl, _ = get_dataloaders(dl_args)

    print('Construct model...')
    n_classes = cfg.SYSTEM.N_CLASSES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model_instance_segmentation(n_classes)
    # move model to the right device
    model.to(device)

    print('Start training...')
    model = training_loop(model, train_dl, device)

    print('Saving the model...')
    torch.save(model.state_dict(), cfg.SYSTEM.MODEL_PATH)


if __name__ == '__main__':
    args = pars_args()
    main(args)
    print('Done.')

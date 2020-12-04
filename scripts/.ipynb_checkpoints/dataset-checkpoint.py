from os.path import join
import torch
import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader


class TinyPascal(Dataset):
    def __init__(self, ann_path, transforms=None, train=True):
        self.ann_path = ann_path
        self.coco = COCO(ann_path) # load training annotations
        self.keys = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.train = train
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        imgIds = self.keys[idx] # Use the key above to retrieve information of the image
        img_info = self.coco.loadImgs(ids=imgIds)

        # Load image
        if self.train:
            data_folder = 'train_images'
        else:
            data_folder = 'test_images'
            
        image = Image.open(join(self.ann_path.split('/')[0], 
                                data_folder, 
                                img_info[0]['file_name']))
        
        target = self._get_target(imgIds)
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def _get_target(self, imgIds):
        annids = self.coco.getAnnIds(imgIds=imgIds)
        anns = self.coco.loadAnns(annids)
        
        if not self.train:
            return anns # []
        
        target = {
            'boxes':[],
            'labels':[],
            'image_id':[],
            'area':[],
            'iscrowd':[],
            'masks':[]
        }
        
        for an in anns:
            target['area'].append(an['area'])
            target['boxes'].append([an['bbox'][0], an['bbox'][1], 
                                    an['bbox'][0]+an['bbox'][2],
                                    an['bbox'][1]+an['bbox'][3]])
            target['labels'].append(an['category_id'])
            target['masks'].append(self.coco.annToMask(an))
            target['iscrowd'].append(an['iscrowd'])
            
        target['image_id'].append(an['image_id'])
        # to tensors
        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        target['image_id'] = torch.tensor(target['image_id'], dtype=torch.int64)
        target['area'] = torch.tensor(target['area'], dtype=torch.int64)
        target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.uint8)
        target['masks'] = torch.tensor(target['masks'], dtype=torch.uint8)
        
        return target

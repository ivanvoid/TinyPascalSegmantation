from detectron2.structures import BoxMode

from pycocotools.coco import COCO
from os.path import join

def get_pascal_dict(ann_path, test=False):
    coco = COCO(ann_path)

    dataset_dicts = []
    data_folder = 'train_images' if not test else 'test_images'
    for idx, key in enumerate(coco.imgs.keys()):
        record = {}

        # Load coco
        img_info = coco.loadImgs(ids=key)

        # Image path
        global_path = join(*ann_path.split('/')[:-1])
        img_path = join(global_path, 
                        data_folder, 
                        img_info[0]['file_name'])

        annids = coco.getAnnIds(imgIds=key)
        anns = coco.loadAnns(annids)

        record["file_name"] = img_path
        record["image_id"] = img_info[0]['id']
        record["height"] = img_info[0]['height']
        record["width"] = img_info[0]['width']

        objs = []
        if not test:
            for an in anns:
                obj = {
                "bbox": [an['bbox'][0], an['bbox'][1], 
                an['bbox'][0]+an['bbox'][2],
                an['bbox'][1]+an['bbox'][3]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": an['segmentation'],
                "category_id": an['category_id'],
                }
                objs.append(obj)
            record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

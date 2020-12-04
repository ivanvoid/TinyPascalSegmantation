# TinyPascalSegmantation
Semantic segmentation for Tiny Pascal dataset.

## Repository structure
<pre>
├── scripts             <- Help files: dataset class and config initializer
├── outputs             <- Where submissiom.json will be stored
├── models              <- Where models will be stored
├── def_utils           <- COCO toolbox
├── data                <- Raw data
├── configs             <- Custum user configs
├── train.py            <- Script for model training
├── test.py             <- Script for inference 
├── README.md           <- This file
├── Report_0860838.pdf  <- Project report
</pre>

## Reproducing steps

### Setup configs
Create new config and put it into *configs/my_new_config.yaml*, structure of config content you can check in *scripts/config.py*.

### Training
```
$ python train.py --confpath configs/ex2.yaml 
```
After training is done you can find model in *models/* folder.

### Inference
```
$ python test.py --confpath configs/ex2.yaml 
```
After inference you can find submission file in *outputs/* folder.

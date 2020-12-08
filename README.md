# TinyPascalSegmantation
Semantic segmentation for Tiny Pascal dataset.

## Repository structure
<pre>
├── data                <- Raw data
├── outputs             <- Where submissiom.json and model will be stored
├── configs             <- Custum user configs
├── train.py            <- Script for model training
├── test.py             <- Script for inference 
├── README.md           <- This file
├── Report_0860838.pdf  <- Project report
</pre>

## Reproducing steps
Install dependencies, then:

### Training
```
$ python train.py
```
After training is done you can find model in *./outputs* folder.

### Inference
```
$ python test.py
```
After inference you can find submission file in *./outputs* folder.

# Dependencies   
- python==3.6.9

- torch, torchvision:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```
- detectron2
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```
- opencv:
```
conda install -c conda-forge opencv
```
- shapely, pyyaml==5.1:
```
conda install shapely, pyyaml==5.1
```
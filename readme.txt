# dependencies:   
python==3.6.9

torch, torchvision:
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

opencv:
conda install -c conda-forge opencv

shapely, pyyaml==5.1:
conda install shapely, pyyaml==5.1
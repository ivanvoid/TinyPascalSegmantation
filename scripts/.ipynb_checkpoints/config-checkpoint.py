# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.DATA_FOLDER = 'data'
# Number of classes in dataset
_C.SYSTEM.N_CLASSES = 21
# Path where save models to
_C.SYSTEM.MODEL_PATH = './models/test_run.pth'


_C.TRAIN = CN()
# Train batch size
_C.TRAIN.BATCH_SIZE = 2
# Number of epochs for training
_C.TRAIN.N_EPOCHS = 1
# Type of loss for training. CE; WCE
_C.TRAIN.LOSS_TYPE = 'CE'
# Type of loss for training. SGD; ADAM
_C.TRAIN.OPTIM_TYPE = 'SGD'


_C.TEST = CN()
# Test batch size
_C.TEST.BATCH_SIZE = 1


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C  # users can `from config import cfg`
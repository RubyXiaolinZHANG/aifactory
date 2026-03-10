from .unet import AinrUnet
from .unet_tuning import AinrUnetWithParamTuning
from .model_ainr_3ds_36dot7G import MainDenoise as example

MODELS = {AinrUnet.__name__: AinrUnet,
          AinrUnetWithParamTuning.__name__: AinrUnetWithParamTuning,
          "example": example}

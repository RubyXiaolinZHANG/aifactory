from .unet import AinrUnet
from .model_ainr_3ds_36dot7G import MainDenoise as example


MODELS = {AinrUnet.__name__: AinrUnet,
          "example": example}
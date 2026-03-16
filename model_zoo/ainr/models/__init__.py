from .unet import AinrUnet
from .unet_tuning import AinrUnetWithParamTuning
from .unet_transformer import AinrUnetTransformer
from .unet_transformer_single import AinrUnetTransformerSingleFrame
from .model_ainr_3ds_36dot7G import MainDenoise as example

MODELS = {AinrUnet.__name__: AinrUnet,
          AinrUnetWithParamTuning.__name__: AinrUnetWithParamTuning,
          AinrUnetTransformer.__name__: AinrUnetTransformer,
          AinrUnetTransformerSingleFrame.__name__: AinrUnetTransformerSingleFrame,
          "example": example}

from aifactory.core import BasicEvaluator
from .ainr_infer import AinrInference


class AinrEvaluator(BasicEvaluator, AinrInference):

    def __init__(self, model, dataloaders, ckpt=None):
        BasicEvaluator.__init__(self, model, dataloaders, ckpt=ckpt)


    def __call__(self, *args, **kwargs):
        pass
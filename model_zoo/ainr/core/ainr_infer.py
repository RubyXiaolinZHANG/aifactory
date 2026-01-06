from aifactory.core import BasicInference

class AinrInference(BasicInference):

    def __init__(self, model, ckpt=None):
        super().__init__(model, ckpt)

import torch
from .basic import PipelineOperator


class BasicInference(PipelineOperator):


    _model = None
    _ckpt = None

    def __init__(self, model, ckpt=None, device=torch.device("cpu"), log=None, **kwargs):
        PipelineOperator.__init__(self, model,
                                  ckpt=ckpt,
                                  device=device,
                                  log=log,
                                  **kwargs)


    def start(self):
        pass


    def finish(self):
        pass

from .basic import PipelineOperator


class BasicInference(PipelineOperator):


    _model = None
    _ckpt = None

    def __init__(self, model, ckpt=None, datas=None):
        super().__init__(model, ckpt, datas)
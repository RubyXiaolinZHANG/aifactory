from .basic import PipelineOperator


class BasicEvaluator():

    def __init__(self, model, dataloaders, ckpt=None):
        PipelineOperator.__init__(model, ckpt, dataloaders)

    def __call__(self, *args, **kwargs):
        pass


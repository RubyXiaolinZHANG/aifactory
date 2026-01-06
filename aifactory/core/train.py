import torch
from .basic import PipelineOperator
from .optimizers import Optimiser


class BasicTrainer(PipelineOperator):
    _evaluator = None
    _optimizer_handler = None
    _optimizer = None
    _iterations = None
    _log_interval = None
    _save_interval = None

    def __init__(self, model, dataloaders, optimizer_params, iterations,
                 evaluator=None,
                 ckpt=None,
                 device=torch.device("cpu"),
                 log=None,
                 log_interval=10,
                 save_interval=1000):
        PipelineOperator.__init__(self, model, ckpt, dataloaders, device, log=log)
        self._evaluator = evaluator
        self.init_optimizers(optimizer_params)
        self._iterations = iterations
        self._log_interval = log_interval
        self._save_interval = save_interval

    def init_optimizers(self, optimizer_params):
        assert isinstance(self._model, torch.nn.Module)
        self._optimizer_handler = Optimiser(optimizer_params, self._model.parameters())
        self._optimizer = self._optimizer_handler.optimizer

    @property
    def optimizer(self):
        return self._optimizer

    def __call__(self, *args, **kwargs):
        pass

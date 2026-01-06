import torch
from aifactory.core import BasicTrainer
from .ainr_infer import AinrInference


class AinrTrainer(BasicTrainer, AinrInference):

    def __init__(self, model, dataloaders, optimizer_params, iterations,
                 device=torch.device("cpu"),
                 ckpt=None,
                 log=None,
                 log_interval=10,
                 save_interval=1000):
        BasicTrainer.__init__(self, model, dataloaders, optimizer_params, iterations,
                              ckpt=ckpt,
                              device=device,
                              log=log,
                              log_interval=log_interval,
                              save_interval=save_interval)

    def __call__(self, *args, **kwargs):
        self._log.info("=" * 50)
        self._log.info("start to simulate a training process")
        self._log.info("=" * 50)

import torch.optim

class Optimiser:

    _optimiser = None
    _init_parameters = None
    def __init__(self, optm_params, model_params):
        self._init_parameters = optm_params
        name = self._init_parameters.get("name")
        optimizer_params = self._init_parameters.get("params")

        self._optimiser = getattr(torch.optim, name)(model_params, **optimizer_params)

    @property
    def optimizer(self):
        return self._optimiser

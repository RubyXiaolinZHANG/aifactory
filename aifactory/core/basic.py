import torch
from aifactory.libs.log.log import ExperimentLogger


class ModelOperator:
    _model = None
    _ckpt = None

    def __init__(self, model, ckpt=None, device=torch.device("cpu")):
        self._model = model.to(device)
        self._ckpt = ckpt
        self._device = device

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    def load_ckpt(self):
        if self._ckpt is not None:
            pass


class PipelineOperator:
    _dataloaders = None
    _device = None
    _model_operator = None
    _model = None
    _log = None

    # update by iteration
    _datas = None
    _inputs = None
    _outputs = None
    _results = None

    def __init__(self, model,
                 ckpt=None,
                 dataloaders=None,
                 device=torch.device("cpu"),
                 log=None):
        self.init_log(log)
        self._device = device
        self.init_model(model, ckpt, device)
        self.init_dataloaders(dataloaders)

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    def init_log(self, log):
        if log is None:
            self._log = print
        else:
            self._log = ExperimentLogger(**log)

    def init_model(self, model, ckpt, device):
        assert isinstance(model, torch.nn.Module)
        self._model_operator = ModelOperator(model, ckpt, device)
        self._model = self._model_operator.model

    def init_dataloaders(self, dataloaders):
        if dataloaders is None:
            return
        assert isinstance(dataloaders, (torch.utils.data.DataLoader,
                                        list,
                                        dict))
        if isinstance(dataloaders, torch.utils.data.DataLoader):
            self._dataloaders = {dataloaders.dataset.__class__.__name__: {'id': 0,
                                                                          "dataloader": dataloaders,
                                                                          "iter": iter(dataloaders)}
                                 }
        elif isinstance(dataloaders, (tuple, list)):
            for db_id, dataloader in enumerate(dataloaders):
                assert isinstance(dataloader, torch.utils.data.DataLoader)
                if self._dataloaders is None:
                    self._dataloaders = {}
                self._dataloaders[dataloader.dataset.__class__.__name__] = {'id': db_id,
                                                                            "dataloader": dataloader,
                                                                            "iter": iter(dataloader)}
        elif isinstance(dataloaders, dict):
            for db_id, (name, dataloader) in enumerate(dataloaders.items()):
                assert isinstance(dataloader, torch.utils.data.DataLoader)
                if self._dataloaders is None:
                    self._dataloaders = {}
                self._dataloaders[name] = {'id': db_id,
                                           "dataloader": dataloader,
                                           "iter": iter(dataloader)
                                           }
        else:
            raise ValueError("Do not support dataloader type: {}".format(type(dataloaders)))
        assert self._dataloaders is not None and len(self._dataloaders) > 0



    def __call__(self, *args, **kwargs):
        self.get_data()
        self.parse_data()
        self.preprocess()
        self.run()
        self.postprocess()

    def get_data(self):
        self._datas = None
        for name, dataset in self._dataloaders.items():
            try:
                data = next(dataset['iter'])
            except StopIteration:
                dataset['iter'] = iter(dataset["dataloader"])
                data = next(dataset['iter'])
            assert data is not None
            if datas is None:
                datas = {}
            datas[name] = data
        return self._datas

    def parse_data(self):
        return self._datas

    def preprocess(self):
        return self._inputs

    def run(self):
        return self._outputs

    def postprocess(self):
        return self._results
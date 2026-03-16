import torch, onnx, os
import builtins
from torchinfo import summary
from aifactory.libs.log.log import ExperimentLogger, TqdmLog


class ModelOperator:
    _model = None
    _model_inputs = None
    _model_input_shapes = None
    _ckpt = None
    _model_info = None
    _log = None
    _verbose = 0
    _dtype = None

    def __init__(self, model, model_inputs=None, model_input_shapes=None,
                 ckpt=None, device=torch.device("cpu"), dtype=torch.float32, log=None):
        self._model = model.to(device)
        if dtype == torch.float16:
            self._model = self._model.half().to(device)
        self._model_inputs = list(model_inputs.values()) if isinstance(model_inputs,dict) else model_inputs
        self._model_input_shapes = list(model_input_shapes.values()) if isinstance(model_input_shapes,dict) else model_input_shapes
        self._ckpt = ckpt
        self._device = device
        self._log = log
        self._device = dtype

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def model_info(self, verbose=0):
        if verbose != self._verbose:
            self._verbose = verbose
            self._model_info = None

        if self._model_info is None:
            if verbose > 0 and isinstance(self._log, ExperimentLogger):
                org_print = print
                self._log.raw_log_enable()
                builtins.print = self._log.info
            self._model_info = summary(self._model,
                                       input_size=self._model_input_shapes,
                                       input_data=self._model_inputs,
                                       verbose=verbose)
            if verbose > 0 and isinstance(self._log, ExperimentLogger):
                self._log.info("\n", raw=True)
                self._log.raw_log_disable()
                builtins.print = org_print
        else:
            self._log.info("\n", raw=True)
            self._log.info(self._model_info, raw=True)
        return self._model_info

    def load_ckpt(self):
        if self._ckpt is not None:
            pass

    def set_log(self, log):
        self._log = log

    def remove_log(self, log):
        self._log = None


class PipelineOperator:
    _dataloaders = None
    _device = None
    _model_operator = None
    _model = None
    _log = None
    _dtype = None
    _save_path = None

    # update by iteration
    _datas = None
    _inputs = None
    _results = None
    _outputs = None

    def __init__(self, model,
                 ckpt=None,
                 dataloaders=None,
                 device=torch.device("cpu"),
                 log=None,
                 **kwargs):
        self._dtype = kwargs.get("dtype", torch.float32)
        self._save_path = kwargs.get("save_path", "./")
        self.init_log(log)
        self._device = device
        self.init_model(model,
                        model_inputs=kwargs.get("model_inputs", None),
                        model_input_shapes=kwargs.get("model_input_shapes", None),
                        ckpt=ckpt,
                        device=device,
                        export_onnx=kwargs.get("export_onnx", None))
        self.init_dataloaders(dataloaders)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def model(self):
        return self._model

    def init_model(self, model, model_inputs, model_input_shapes, ckpt, device, export_onnx=None, verbose=0):
        assert isinstance(model, torch.nn.Module)
        self._model_operator = ModelOperator(model,
                                             model_inputs=model_inputs,
                                             model_input_shapes=model_input_shapes,
                                             ckpt=ckpt,
                                             device=device,
                                             log=self._log)
        self._model = self._model_operator.model
        self._model_operator.model_info(verbose)
        if ckpt is not None:
            self.ensemble_model_parameters(ckpt)
        else:
            message = "model is not initialized in Pipeline"
            print(message) if self._log is None else self._log.warning(message)
        if export_onnx:
            if model_inputs is not None:
                if isinstance(model_inputs, dict):
                    model_inputs = tuple(model_inputs.values())
                    model_input_names = tuple(model_inputs.keys())
                elif isinstance(model_inputs, list):
                    model_inputs = tuple(model_inputs)
                    model_input_names = None
                elif isinstance(model_inputs, (tuple, torch.Tensor)):
                    model_input_names = None
                else:
                    raise ValueError(
                        "model inputs should be dict, list, tuple or tensor, but the given inputs is {}".format(
                            type(model_inputs)))
                os.makedirs(os.path.dirname(export_onnx))
                torch.onnx.export(self._model, model_inputs, export_onnx, input_names=model_input_names)
                onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(export_onnx)), export_onnx)
            elif model_input_shapes is not None:
                if isinstance(model_input_shapes, dict):
                    model_input_names = tuple(model_input_shapes.keys())
                    model_inputs = tuple([torch.randn(size, device=device) for size in model_input_shapes.values()])
                elif isinstance(model_input_shapes, (tuple, list)):
                    model_input_names = None
                    model_inputs = tuple([torch.randn(size, device=device) for size in model_input_shapes])
                else:
                    raise ValueError(
                        "model inputs should be dict, list, tuple or tensor, but the given inputs is {}".format(
                            type(model_inputs)))
                os.makedirs(os.path.dirname(export_onnx),exist_ok=True)
                torch.onnx.export(self._model, model_inputs, export_onnx, input_names=model_input_names)
                onnx.save(onnx.shape_inference.infer_shapes(onnx.load_model(export_onnx)), export_onnx)
            else:
                info = "Input information is not provided, can not export onnx!"
                if self._log is None:
                    print(info)
                elif isinstance(self._log, TqdmLog):
                    self._log.info(info)
                elif isinstance(self._log, ExperimentLogger):
                    self._log.warning(info)
                else:
                    raise ValueError("Unrecognized type of log: {}".format(type(self._log)))
        else:
            pass

    def set_model(self, model):
        assert isinstance(model, torch.nn.Module)
        self._model = model

    def ensemble_model_parameters(self, ckpt):
        weights = torch.load(ckpt, weights_only=-True)
        if "model" in weights:
            weights = weights["model"]
        self._model.load_state_dict(weights)
        message = "Loading weights succeeded from: {}".format(ckpt)
        print(message) if self._log is None else self._log.info(message)

    def init_dataloaders(self, dataloaders):

        def apply_log_to_dataset(_dataloader):
            if self._log is not None and hasattr(_dataloader.dataset, "set_log"):
                _dataloader.dataset.set_log(self._log)

        if dataloaders is None:
            return
        assert isinstance(dataloaders, (torch.utils.data.DataLoader,
                                        list,
                                        dict))

        if isinstance(dataloaders, torch.utils.data.DataLoader):
            apply_log_to_dataset(dataloaders)
            self._dataloaders = {dataloaders.dataset.__class__.__name__: {'id': 0,
                                                                          "dataloader": dataloaders,
                                                                          "iter": iter(dataloaders)}
                                 }
        elif isinstance(dataloaders, (tuple, list)):
            for db_id, dataloader in enumerate(dataloaders):
                assert isinstance(dataloader, torch.utils.data.DataLoader)
                if self._dataloaders is None:
                    self._dataloaders = {}
                apply_log_to_dataset(dataloader)
                self._dataloaders[dataloader.dataset.__class__.__name__] = {'id': db_id,
                                                                            "dataloader": dataloader,
                                                                            "iter": iter(dataloader)}
        elif isinstance(dataloaders, dict):
            for db_id, (name, dataloader) in enumerate(dataloaders.items()):
                assert isinstance(dataloader, torch.utils.data.DataLoader)
                if self._dataloaders is None:
                    self._dataloaders = {}
                apply_log_to_dataset(dataloader)
                self._dataloaders[name] = {'id': db_id,
                                           "dataloader": dataloader,
                                           "iter": iter(dataloader)
                                           }
        else:
            raise ValueError("Do not support dataloader type: {}".format(type(dataloaders)))
        assert self._dataloaders is not None and len(self._dataloaders) > 0

    def close_dataloaders(self):
        if self._dataloaders is None:
            return
        for name, dataset in self._dataloaders.items():
            if hasattr(dataset["dataloader"].dataset, "finish"):
                dataset["dataloader"].dataset.finish()

    def init_log(self, log):
        if log is None:
            self._log = TqdmLog()
        else:
            self._log = ExperimentLogger(**log)

    def set_log(self, log):
        assert isinstance(log, TqdmLog) or isinstance(log, ExperimentLogger)
        self._log = log

    def remove_log(self):
        if isinstance(self._log, ExperimentLogger):
            self._log.finish()
        else:
            self._log = None

    def sync_log(self, log):
        self._log = log
        self._model_operator._log = log

    def __call__(self, *args, **kwargs):
        # some log and tasks i
        self.start()
        result = self.run(*args, **kwargs)
        self.finish()
        return result

    def start(self):
        if self._model is not None:
            assert isinstance(self._model, torch.nn.Module)
            self._model_operator.model_info(verbose=2)

    def run(self, *args, **kwargs):
        pass

    def finish(self):
        self.close_dataloaders()
        self.remove_log()

    def get_data(self):
        self._datas = None
        for name, dataset in self._dataloaders.items():
            try:
                data = next(dataset['iter'])
            except StopIteration:
                dataset['iter'] = iter(dataset["dataloader"])
                data = next(dataset['iter'])
            assert data is not None
            if self._datas is None:
                self._datas = {}
            self._datas[name] = data
        return self._datas

    def model_infer(self):
        if isinstance(self._inputs, torch.Tensor):
            self._outputs = self._model(self._inputs)
        elif isinstance(self._inputs, (tuple, list)):
            self._outputs = self._model(*self._inputs)
        elif isinstance(self._inputs, dict):
            self._outputs = self._model(**self._inputs)
        else:
            raise ValueError("Do not support inputs type of {}".format(type(self._inputs)))
        return self._outputs

    def parse_data(self):
        pass

    def preprocess(self):
        pass

    def postprocess(self):
        pass
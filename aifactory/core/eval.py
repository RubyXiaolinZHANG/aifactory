from copyreg import add_extension

import torch
from .basic import PipelineOperator
from aifactory.libs.metrics import METRICS


class BasicEvaluator(PipelineOperator):
    _metrics = None
    _dominate_metrics = None
    _log_interval = None
    _training_iter = None
    _count = None

    def __init__(self, model, dataloaders, metrics_params, dominate_metrics=None, ckpt=None,
                 device=torch.device("cpu"),
                 log=None,
                 log_interval=10,
                 **kwargs):
        PipelineOperator.__init__(self, model,
                                  ckpt=ckpt,
                                  dataloaders=dataloaders,
                                  device=device,
                                  log=log,
                                  **kwargs)
        self._dominate_metrics = dominate_metrics.lower() if dominate_metrics is not None else list(metrics_params.keys())[0].lower()
        self._log_interval = log_interval
        self._count = 0
        self.init_metrics(metrics_params)

    def set_training_iter(self, iter):
        self._training_iter = iter

    def init_metrics(self, metrics_params):
        for name, parameters in metrics_params.items():
            if not parameters.get("enable", False):
                continue
            if self._metrics is None:
                self._metrics = {}
            func = METRICS[name]() if parameters.get("params") is None else METRICS[name](**parameters.get("params"))
            self._metrics[name] = {"name": name,
                                   "func": func}

    def start(self):
        self._log.info("=" * 100)
        self._log.info("start evaluating!")
        self._log.info("=" * 100)
        if self._training_iter is None or self._training_iter == 0:
            PipelineOperator.start(self)
        self._log.info("=" * 100)
        self._log.info("{:^90}".format("evaluate databases"))
        self._log.info(
            "{:<5} {:<15} {:<15} {:<15} {:<10}".format("ID", "Name", "Total Samples", "Batch Size", "Iterations"))
        self._log.info("=" * 100)
        if self._dataloaders is not None:
            for name, database in self._dataloaders.items():
                self._log.info("{:<5} {:<15} {:<15} {:<15} {:<10}".format(database['id'],
                                                                          name,
                                                                          len(database['dataloader'].dataset),
                                                                          database['dataloader'].batch_size,
                                                                          len(database['dataloader'])))
        self._log.info("=" * 100)

    def run(self):
        for name, dataloader in self._dataloaders.items():
            self._log.info("=" * 100)
            self._log.info("start evaluating: {}".format(name))
            self._log.info("=" * 100)
            for batch_id, batch in enumerate(dataloader["dataloader"]):
                self._datas = {name: batch}
                self.parse_data()
                self.preprocess()
                # self.eval()
                self.model_infer()
                self.postprocess()
                self.compute_metrics()
                self._count += 1

    def prepare_metrics(self):
        pass

    def compute_metrics(self):
        _metrics = self.prepare_metrics()
        for name, metrics in _metrics.items():
            if isinstance(metrics["input"], (list, tuple)):
                metrics["value"] = metrics["func"](*metrics["input"],
                                                   **metrics["params"] if metrics.get("params") is not None else {})
            elif isinstance(metrics["input"], dict):
                metrics["value"] = metrics["func"](**metrics["input"],
                                                   **metrics["params"] if metrics.get("params") is not None else {})
            else:
                raise ValueError("loss inputs should be tuple, list, or dict")
        if self._results is None:
            self._results = {}
        self._results["metrics"] = _metrics
        return

import os
from datetime import datetime
import torch

from .basic import PipelineOperator
from .optimizers import Optimiser
from aifactory.utils.nan_debugger import NanDebugger
from aifactory.utils.monitor import ModuleMonitor
from aifactory.libs.loss import LOSSES


class BasicTrainer(PipelineOperator):
    _optimizer_handler = None
    _optimizer = None
    _iterations = None
    _log_interval = None
    _save_interval = None
    _evaluator = None
    _evaluate_before_train = False
    _losses = None

    # process control
    _count = 0

    # pass parameters
    _gt = None
    _eval_result = None

    # debugger
    _nan_debugger = None
    _monitor = None

    def __init__(self, model, dataloaders, optimizer_params, losses_params, iterations,
                 evaluator=None,
                 ckpt=None,
                 device=torch.device("cpu"),
                 log=None,
                 log_interval=10,
                 save_interval=1000,
                 evaluate_before_train=False,
                 **kwargs):

        PipelineOperator.__init__(self, model,
                                  ckpt=ckpt,
                                  dataloaders=dataloaders,
                                  device=device,
                                  log=log,
                                  **kwargs)
        self._evaluator = evaluator
        self._iterations = iterations
        self._log_interval = log_interval
        self._save_interval = save_interval
        self._evaluate_before_train = evaluate_before_train
        self.init_optimizers(optimizer_params)
        self.init_losses(losses_params)
        self.sync_evaluator()
        self._nan_debugger = NanDebugger(self._model, self._optimizer, os.path.join(self._save_path, "Error"))

    def init_optimizers(self, optimizer_params):
        assert isinstance(self._model, torch.nn.Module)
        self._optimizer_handler = Optimiser(optimizer_params, self._model.parameters())
        self._optimizer = self._optimizer_handler.optimizer

    def init_losses(self, loss_params):
        for name, params in loss_params.items():
            if params.get("enable", True):
                assert name in LOSSES
                if self._losses is None:
                    self._losses = {}
                if "at_iter" in params and params["at_iter"].get("start") is not None:
                    start_iter = params["at_iter"]["start"]
                else:
                    start_iter = 0
                if "at_iter" in params and params["at_iter"].get("end") is not None:
                    end_iter = params["at_iter"]["end"]
                else:
                    end_iter = self._iterations
                if params.get("params") is None:
                    func = LOSSES[name]()
                else:
                    func = LOSSES[name](**params.get("params"))
                self._losses[name] = {"func": func,
                                      "start_iter": start_iter,
                                      "end_iter": end_iter,
                                      "weight": params.get("weight", 1.0),
                                      "init_parameters": params}

    def sync_evaluator(self):
        if self._evaluator is None:
            return
        if self._log is not None:
            self._evaluator.sync_log(self._log)

    @property
    def optimizer(self):
        return self._optimizer

    def start(self):

        if self._evaluate_before_train and self._evaluator is not None:
            self._log.info("=" * 100)
            self._log.info("perform evaluating before training!")
            self._log.info("=" * 100)
            self._evaluator.set_training_iter(0)
            self._eval_result = self._evaluator()
        else:
            self._log.info("=" * 100)
            self._log.info("model information")
            self._log.info("=" * 100)
            PipelineOperator.start(self)

        self._log.info("=" * 100)
        self._log.info("start training!")
        self._log.info("=" * 100)
        self._log.info("=" * 100)
        self._log.info("{:^90}".format("training databases"))
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
        while self._count < self._iterations:
            self.get_data()
            self.parse_data()
            self.preprocess()
            self.train()
            self.postprocess()
            self._count += 1

    def train(self):
        is_nan = False

        # forward
        self.model_infer()
        nan_info = self.forward_nan_detection()
        if nan_info is not None:
            is_nan = True
            self._log.error("{}".format("=" * 100))
            self._log.error("{:^100}".format("ITER {}: NaN detected in model outputs!".format(self._count)))
            self._log.error("{}".format("=" * 100))
            self.print_nan_info(nan_info)
            self._nan_debugger.save_snapshot(self._count)

        # compute losses
        if not is_nan:
            self.compute_losses()
            nan_info, losses = self.loss_nan_detection()
            if nan_info is not None:
                is_nan = True
                self._log.error("{}".format("=" * 100))
                self._log.error("{:^100}".format("ITER {}: NaN detected in loss!".format(self._count)))
                self._log.error("{}".format("=" * 100))
                self.print_nan_info(nan_info)
                self._nan_debugger.save_snapshot(self._count, losses)

        # backward
        if not is_nan:
            self.backward()
            nan_info = self.backward_nan_detection()
            if nan_info is not None:
                is_nan = True
                self._log.error("{}".format("=" * 100))
                self._log.error("{:^100}".format("ITER {}: NaN detected in gradient!".format(self._count)))
                self._log.error("{}".format("=" * 100))
                self.print_nan_info(nan_info)
                self._nan_debugger.save_snapshot(self._count, losses)

        if is_nan:
            self._log.error("Error snapshot is saved into {}".format(self._nan_debugger.snapshot_dir))
            # self.save_present_state(self._nan_debugger.snapshot_dir)
            exit()
        # self.save_present_state(self._nan_debugger.snapshot_dir)
        # update parameters
        self._optimizer.step()

    def prepare_losses(self):
        pass

    def compute_losses(self):
        losses = self.prepare_losses()
        total_loss = None
        for name, loss in losses.items():
            if isinstance(loss["input"], (list, tuple)):
                loss_ = loss["func"](*loss["input"], **loss["params"] if loss.get("params") is not None else {})
            elif isinstance(loss["input"], dict):
                loss_ = loss["func"](**loss["input"], **loss["params"] if loss.get("params") is not None else {})
            else:
                raise ValueError("loss inputs should be tuple, list, or dict")
            if isinstance(loss_, torch.Tensor):
                loss["value"] = loss_
            elif isinstance(loss_, tuple) and isinstance(loss_[1], dict):
                loss["value"] = loss_[0]
                loss["components"] = loss_[1]
            else:
                raise ValueError("type of loss should be tensor or tuple[2]: (tensor, dict).")
            if total_loss is None:
                total_loss = loss["value"] * loss["weight"]
            else:
                total_loss += loss["value"] * loss["weight"]
        losses["total"] = total_loss
        if self._results is None:
            self._results = {}
        self._results['losses'] = losses
        return losses

    def backward(self):
        self._optimizer.zero_grad()
        self._results['losses']["total"].backward()
        # self._optimizer.step()

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self._model.state_dict(), save_path)

    def nan_detection(self, data):
        info = None
        if isinstance(data, dict):
            # check net outputs
            for name, output in data.items():
                if torch.isnan(output).any():
                    if info is None:
                        info = {}
                        info[name] = {"is_nan": True,
                                      "info": "NaN detected in {}! MIN={:.6f}, MAX={:.6f}".format(name,
                                                                                                  output.min(),
                                                                                                  output.max())}
                elif torch.isinf(output).any():
                    if info is None:
                        info = {}
                        info[name] = {"is_inf": True,
                                      "info": "Inf detected in {}! MIN={:.6f}, MAX={:.6f}".format(name,
                                                                                                  output.min(),
                                                                                                  output.max())}
                else:
                    pass
        elif isinstance(data, (tuple, list)):
            for output_id, output in enumerate(data):
                if torch.isnan(output).any():
                    if info is None:
                        info = {}
                        info[output_id] = {"is_nan": True,
                                           "info": "NaN detected in tuple[{}]! MIN={:.6f}, MAX={:.6f}".format(output_id,
                                                                                                              output.min(),
                                                                                                              output.max())}
                elif torch.isinf(output).any():
                    if info is None:
                        info = {}
                        info[output_id] = {"is_inf": True,
                                           "info": "Inf detected in tuple[{}]! MIN={:.6f}, MAX={:.6f}".format(output_id,
                                                                                                              output.min(),
                                                                                                              output.max())}
                else:
                    pass
        elif isinstance(data, torch.Tensor):
            if torch.isnan(self._outputs).any():
                if info is None:
                    info = {}
                    info["tensor"] = {"is_nan": True,
                                      "info": "NaN detected in Tensor! MIN={:.6f}, MAX={:.6f}".format(
                                          self._outputs.min(),
                                          self._outputs.max())}
            elif torch.isinf(self._outputs).any():
                if info is None:
                    info = {}
                    info["tensor"] = {"is_inf": True,
                                      "info": "Inf detected in Tensor! MIN={:.6f}, MAX={:.6f}".format(
                                          self._outputs.min(),
                                          self._outputs.max())}
            else:
                pass
        else:
            pass
        return info

    def forward_nan_detection(self):
        info = self.nan_detection(self._outputs)
        if info is not None:
            info["current_state"] = {"inputs": self._inputs,
                                     "outputs": self._outputs}
        return info

    def loss_nan_detection(self):
        losses = {}
        for name, loss in self._results['losses'].items():
            if isinstance(loss, torch.Tensor):
                losses[name] = loss
            elif isinstance(loss, dict):
                losses[name] = loss["value"]
            else:
                raise ValueError("Do not support loss type of {}".format(type(loss)))
        loss_info = self.nan_detection(losses)
        if loss_info is not None:
            loss_info["current_state"] = {"inputs": self._inputs,
                                          "outputs": self._outputs,
                                          "losses": self._results['losses']}
        return loss_info, losses

    def backward_nan_detection(self):
        gradients = None
        for name, param in self._model.named_parameters():
            if param.grad is None:
                continue
            if gradients is None:
                gradients = {}
            gradients[name] = param.grad
        info = self.nan_detection(gradients)
        if info is not None:
            info["current_state"] = {"inputs": self._inputs,
                                     "outputs": self._outputs,
                                     "losses": self._results['losses']}
        return info

    def print_nan_info(self, nan_info):
        for id, (name, info) in enumerate(nan_info.items()):
            self._log.info("[{}/{}]\t{}".format(id, len(nan_info), info["info"]))

    def save_present_state(self, save_dir):
        self._monitor = self.init_monitor()

        # repeat forward
        org_outputs = self._outputs
        self.model_infer()
        self.compute_losses()

        # repeat backward
        self.backward()
        print("**************")

        # repeat backward
        save_dir = os.path.join(save_dir, "Error_NaN_{}".format(datetime.now().strftime('%Y%m%d_%H%M')))
        os.makedirs(save_dir, exist_ok=True)
        torch.save({"inputs": self._inputs,
                    "outputs": self._outputs,
                    "targets": self._gt,
                    "model": self._model.state_dict(),
                    "optmizer": self._optimizer.state_dict(),
                    }
                   )

    def init_monitor(self):

        target_modules = None
        for name, loss in self._losses.items():
            if isinstance(loss['func'], torch.nn.Module):
                if loss['func'].__class__.__name__ == "GradientMagnitudePhaseLoss":
                    continue
                loss.name = name
                if target_modules is None:
                    target_modules = {}
                target_modules[name] = loss['func']

        return ModuleMonitor(target_modules)

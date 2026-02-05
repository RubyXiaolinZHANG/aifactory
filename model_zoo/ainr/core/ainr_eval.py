import os
import torch
import cv2
from copy import deepcopy
from tqdm import tqdm
from aifactory.core import BasicEvaluator
from .ainr_infer import AinrInference


class AinrEvaluator(BasicEvaluator, AinrInference):
    _vis = None
    _result_collection = None
    _result_statistics = None

    def __init__(self, model, dataloaders, metrics_params,
                 dominate_metrics=None,  # default is the first metrics
                 ckpt=None,
                 device=torch.device("cpu"),
                 log=None,
                 log_interval=10,
                 **kwargs):
        BasicEvaluator.__init__(self, model, dataloaders, metrics_params,
                                dominate_metrics=dominate_metrics,
                                ckpt=ckpt,
                                device=device,
                                log=log,
                                log_interval=log_interval,
                                **kwargs)
        self._vis = kwargs.get("vis", None)

    def run(self):
        for name, dataloader in self._dataloaders.items():
            self._log.info("start evaluating: {}".format(name))
            if self._vis is not None and self._vis.get("save2local", False):
                save_dir = os.path.join(self._save_path, "images",
                                        name) if self._training_iter is None else os.path.join(self._save_path,
                                                                                               "iter_{:08d}".format(
                                                                                                   self._training_iter),
                                                                                               "images", name)
                self._log.info("evaluating results save to: {}".format(save_dir))
            self._count = 0
            batch_num = len(dataloader["dataloader"])
            with tqdm(total=batch_num, desc="TEST DATABASE |{:^30}|".format(name)) as pbar:
                for batch_id, batch in enumerate(dataloader["dataloader"]):
                    self._datas = {name: batch}
                    self.parse_data()
                    with torch.no_grad():
                        self.eval_video()
                    self.update_eval_results()
                    self._vis["stop"] = self._count > (batch_num * self._vis.get("save_ratio", 1.0))
                    self.save()
                    self._count += 1
                    pbar.update(1)
                    if self._count == -1:
                        break
            tqdm.write("")
            self.process_result(name)
        self.log()
        return self._result_statistics

    def eval_video(self):
        self._frame_id = 0
        while True:
            if self.get_frame() is None:
                break
            self.preprocess()
            self.model_infer()
            self.postprocess(enable_raw2bgr=True)
            self.compute_metrics()
            self._frame_id += 1

    def get_frame(self):
        sensor_raw = []
        gt = []
        read = []
        shot = []
        black_level = []
        maximum = []
        frame_info = []
        max_h = 0
        max_w = 0
        for db_name, database in self._datas.items():
            batch_size, frames, h, w = database['sensor_raw'].shape
            max_h = max_h if max_h > h else h
            max_w = max_w if max_w > w else w
            if self._frame_id < frames:
                sensor_raw.append(database['sensor_raw'][:, self._frame_id].unsqueeze(dim=1))
                gt.append(database['gt'][:, self._frame_id].unsqueeze(dim=1))
                read.append(database["cam"]["read"])
                shot.append(database["cam"]["shot"])
                black_level.append(database["cam"]['black_level'])
                maximum.append(database["cam"]['maximum'])
                frame_info.append({"database": db_name,
                                   "frame_id": self._frame_id,
                                   "frame_num": batch_size})
        if len(sensor_raw) > 0:
            this_batch_frames = {"sensor_raw": torch.concat(sensor_raw, dim=0),
                                 "clean_raw": torch.concat(gt, dim=0),
                                 "read": torch.concat(read),
                                 "shot": torch.concat(shot),
                                 'black_level': torch.concat(black_level),
                                 "maximum": torch.concat(maximum),
                                 "frame_info": frame_info}
            if self._frame_sequence is None:
                self._frame_sequence = {}
            self._frame_sequence[self._frame_id] = this_batch_frames
            return this_batch_frames
        else:
            return None

    def prepare_metrics(self):
        this_batch_frames = self._frame_sequence[self._frame_id]
        metrics = None
        for name, _metrics in self._metrics.items():
            if name in ["psnr", "pixel_l1", "channel_l1", "mse"]:
                if metrics is None:
                    metrics = {}
                metrics[name] = deepcopy(_metrics)
                metrics[name]["input"] = [this_batch_frames['bgr_fusion'].to(torch.float32),
                                          this_batch_frames['bgr_gt'].to(torch.float32)]
        assert metrics is not None
        return metrics

    def compute_metrics(self):
        super().compute_metrics()
        self._results[self._frame_id]["metrics"] = self._results.pop("metrics")

    def update_eval_results(self):
        if self._result_collection is None:
            self._result_collection = {}
        # metrics_name = list(self._metrics.keys())
        for name, data in self._datas.items():
            if name not in self._result_collection:
                self._result_collection[name] = {}
            for sample_id, sample_name in enumerate(data['sample_name']):
                if sample_name not in self._result_collection[name]:
                    self._result_collection[name][sample_name] = {}
                for frame_id in range(data['sequence_length'][sample_id]):
                    for key, metrics in self._results[frame_id]['metrics'].items():
                        if key not in self._result_collection[name][sample_name]:
                            self._result_collection[name][sample_name][key] = []
                        if metrics['value'].shape[0] == len(data['sample_name']):
                            val = metrics['value'][sample_id]
                        elif metrics['value'].numel() == 1:
                            val = metrics['value']
                        else:
                            raise ValueError("shape of metrics {} is {}, but sample num is {}".format(key,
                                                                                                      metrics[
                                                                                                          'value'].shape,
                                                                                                      len(data[
                                                                                                              'sample_name'])))
                        self._result_collection[name][sample_name][key].append(val)

    def log(self):
        if self._log is None:
            return
        if self._log.__class__.__name__ != "ExperimentLogger":
            return
        metrics = {}
        target_metrics_item = ["mean", "std", 'skewness', 'kurtosis']
        for database_id, (database_name, database_result) in enumerate(self._result_statistics.items()):
            for metrics_name, metrics_value in database_result.items():
                for name_id, name in enumerate(target_metrics_item):
                    if name in metrics_value:
                        if isinstance(metrics_value[name], torch.Tensor) and metrics_value[name].numel() == 1:
                            metrics[
                                "eval/{}/{}_{}/{}_{}".format(metrics_name, name_id, name, database_id, database_name)] = \
                                metrics_value[name].item()
                        elif isinstance(metrics_value[name], float):
                            metrics[
                                "eval/{}/{}_{}/{}_{}".format(metrics_name, name_id, name, database_id, database_name)] = \
                                metrics_value[name]
                        else:
                            pass

        if self._training_iter is None:
            self._log.log_metrics(metrics, 0)
        else:
            self._log.log_metrics(metrics, self._training_iter)

    def map_result_to_data(self):

        for sequence_id, frames in self._frame_sequence.items():
            frame_info = frames['frame_info']
            start_id = 0
            for database in frame_info:
                db_name = database['database']
                end_id = start_id + database['frame_num']
                if "bgr_raw" not in self._datas[db_name]:
                    self._datas[db_name]["bgr_raw"] = []
                self._datas[db_name]["bgr_raw"].append(frames['bgr_raw'][start_id:end_id].unsqueeze(dim=1))
                if "bgr_fusion" not in self._datas[db_name]:
                    self._datas[db_name]["bgr_fusion"] = []
                self._datas[db_name]["bgr_fusion"].append(frames['bgr_fusion'][start_id:end_id].unsqueeze(dim=1))
                if 'bgr_gt' not in self._datas[db_name]:
                    self._datas[db_name]['bgr_gt'] = []
                self._datas[db_name]['bgr_gt'].append(frames['bgr_gt'][start_id:end_id].unsqueeze(dim=1))
                start_id = end_id

        target_names = ["bgr_raw", "bgr_fusion", "bgr_gt"]
        for db_name, data in self._datas.items():
            for name in target_names:
                data[name] = torch.concat(data[name], dim=1)

    def save(self):
        self.map_result_to_data()
        cvt_rgb_names = ["bgr_raw", 'bgr_gt', 'bgr_fusion']
        for name, data in self._datas.items():
            for sample_id, sample_name in enumerate(data['sample_name']):
                key = ("eval/{}".format(sample_name)
                       if self._training_iter is None
                       else "eval/ITER:{}/{}".format(self._training_iter, sample_name))
                image_names = [os.path.basename(batch_files[sample_id]).split(".")[0] for batch_files in data["files"]]
                for target_id, target_name in enumerate(cvt_rgb_names):
                    target_key = "{}/{}_{}".format(key, target_id, target_name)
                    sample = data[target_name][sample_id]
                    frame_names = []
                    sample_list = []
                    for frame_id, bgr in enumerate(sample):
                        frame_names.append("seq{}_{}".format(frame_id,
                                                             image_names[data['frame_id'][sample_id, frame_id]]))
                        sample_list.append(cv2.cvtColor(bgr.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB))
                    sample_dict = dict(zip(frame_names, sample_list))

                    if self._vis.get("stop", False):
                        continue
                    # save images to trackio .db
                    if self._vis.get("save2db", False):
                        self._log.log_image_sequence(sample_dict, key=target_key)
                    # save to local folder
                    if self._vis.get("save2local", False):
                        self.save_images(sample_dict,
                                         save_dir=(os.path.join(self._save_path, "images", name)
                                                   if self._training_iter is None
                                                   else os.path.join(self._save_path,
                                                                     "iter_{:08d}".format(self._training_iter),
                                                                     "images", name)),
                                         key=target_key.replace(":", "-").replace("/", "_"),
                                         rgb2bgr=True)

    def summarize_metrics(self, database_name):

        self._log.info("{:^100}".format(database_name))
        self._log.info(
            "{:<15} {:<15} {:<15} {:<15} {:<15}".format("METRICS", "MEAN", "STD", "Skewness", "Kurtosis"))
        self._log.info("-" * 100)
        database_metrics = self._result_statistics[database_name]
        for metrics_name, metrics_val in database_metrics.items():
            if isinstance(metrics_val['mean'], torch.Tensor):
                self._log.info("{:<15} {:<15} {:<15} {:<15} {:<15}".format(metrics_name,
                                                                           "{:.8f}".format(
                                                                               metrics_val['mean'].item()),
                                                                           "{:.8f}".format(
                                                                               metrics_val['std'].item()),
                                                                           "{:.8f}".format(
                                                                               metrics_val['skewness'].item()),
                                                                           "{:.8f}".format(
                                                                               metrics_val['kurtosis'].item())))
            else:
                for i in range(len(metrics_val['mean'])):
                    self._log.info("{:<15} {:<15} {:<15} {:<15} {:<15}".format(metrics_name if i == 0 else "",
                                                                               "{:.8f}".format(
                                                                                   metrics_val['mean'][-1].item()),
                                                                               "{:.8f}".format(
                                                                                   metrics_val['std'][-1].item()),
                                                                               "{:.8f}".format(
                                                                                   metrics_val['skewness'][
                                                                                       -1].item()),
                                                                               "{:.8f}".format(
                                                                                   metrics_val['kurtosis'][
                                                                                       -1].item())))
        # database_metrics["dominate_metrics"] = database_metrics[self._dominate_metrics]
        self._log.info("-" * 100)

    def process_result(self, database_name=None):
        if database_name is None:
            for _database_name in self._result_collection.keys():
                self.process_result(_database_name)
        else:
            assert database_name in self._result_collection
            if self._result_statistics is None:
                self._result_statistics = {}
            self._result_statistics[database_name] = {}

            for metrics_name, metrics_val in self._metrics.items():
                self._result_statistics[database_name][metrics_name] = {"data": []}
            database_metrics = self._result_collection[database_name]
            for sample_name, sample_metrics in database_metrics.items():
                for metrics_name, metrics_val in sample_metrics.items():
                    self._result_statistics[database_name][metrics_name]["data"] += metrics_val
            for metrics_name, metrics_val in self._result_statistics[database_name].items():
                data = (torch.stack(metrics_val['data'], dim=0)
                        if metrics_val['data'][0].numel() == 1
                        else torch.vstack(metrics_val['data']))
                metrics_val['data'] = data
                if data.ndim == 1:
                    hist_density, bin_edges = torch.histogram(data.cpu(), bins=2048, density=True)
                    metrics_val['bin_edges'] = bin_edges
                    metrics_val['hist_density'] = hist_density
                    metrics_val['mean'] = data.mean()
                    metrics_val['std'] = data.std()
                    norm = (data - metrics_val['mean']) / metrics_val['std']
                    metrics_val['skewness'] = (norm ** 3).mean()
                    metrics_val['kurtosis'] = (norm ** 4).mean() - 3
                else:
                    metrics_val['bin_edges'] = []
                    metrics_val['hist_density'] = []
                    metrics_val['mean'] = []
                    metrics_val['std'] = []
                    metrics_val['skewness'] = []
                    metrics_val['kurtosis'] = []
                    metrics_val['data'] = data
                    for i in range(data.shape[-1]):
                        hist_density, bin_edges = torch.histogram(data[:, i].cpu(), bins=2048, density=True)
                        metrics_val['bin_edges'].append(bin_edges)
                        metrics_val['hist_density'].append(hist_density)
                        metrics_val['mean'].append(data[:, i].mean())
                        metrics_val['std'].append(data[:, i].std())
                        norm = (data[:, i] - metrics_val['mean'][i]) / metrics_val['std'][i]
                        metrics_val['skewness'].append((norm ** 3).mean())
                        metrics_val['kurtosis'].append((norm ** 4).mean() - 3)
        self.summarize_metrics(database_name)

    def finish(self):
        self._log.info("=" * 100)
        self._log.info("{:^100}".format("evaluation result"))
        self._log.info("=" * 100)
        for database_name in self._result_statistics.keys():
            self.summarize_metrics(database_name)

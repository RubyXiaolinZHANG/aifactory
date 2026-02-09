import os.path
from copy import deepcopy

import cv2
import numpy as np
import torch
from aifactory.core import BasicTrainer
from aifactory.libs.nn.functionals import vst, ivst, normalization
from aifactory.libs.common.raw2rgb import raw2bgr
from aifactory.utils.container import BoundedMaxHeapDictList
from .ainr_infer import AinrInference


class ROI_NCHW:
    _x, _y, _h, _w = None, None, None, None
    _x1, _y1, _x2, _y2 = None, None, None, None

    def __init__(self, x, y, w, h):
        self._x, self._y, self._h, self._w = x, y, h, w
        self._x1, self._y1, self._x2, self._y2 = x, y, x + w, y + h

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def h(self):
        return self._h

    @property
    def w(self):
        return self._w

    @property
    def x1(self):
        return self._x1

    @property
    def y1(self):
        return self._y1

    @property
    def x2(self):
        return self._x2

    @property
    def y2(self):
        return self._y2

    def crop(self, data):
        return data[:, :, self._y1:self._y2, self._x1:self._x2]


def shape_align(data, dst_h, dst_w):
    n, c, h, w = data.shape
    new_data = torch.zeros((n, c, dst_h, dst_w), dtype=data.dtype, device=data.device)
    new_data[:, :, :h, :w] = data
    return new_data


class AinrTrainer(BasicTrainer, AinrInference):
    _frame_id = 0
    _vis = None

    # save ckpt
    _reserve_best_ckpt = 10
    _best_ckpts = None

    def __init__(self, model, dataloaders, optimizer_params, loss_params, iterations,
                 evaluator=None,
                 device=torch.device("cpu"),
                 ckpt=None,
                 log=None,
                 log_interval=10,
                 save_interval=1000,
                 evaluate_before_train=False,
                 **kwargs):
        BasicTrainer.__init__(self, model, dataloaders, optimizer_params, loss_params, iterations,
                              evaluator=evaluator,
                              ckpt=ckpt,
                              device=device,
                              log=log,
                              log_interval=log_interval,
                              save_interval=save_interval,
                              evaluate_before_train=evaluate_before_train,
                              **kwargs)
        self._count = 0
        self._vis = kwargs.get("vis", None)
        self._best_ckpts = BoundedMaxHeapDictList(self._reserve_best_ckpt, "score")
        self.sync_evaluator()

    def start(self):
        super().start()
        if self._eval_result is None:
            return
        else:
            eval_score, best_ckpt_is_small = self.get_model_eval_parameters()
            self.update_best_ckpt(eval_score, self._ckpt, best_ckpt_is_small=best_ckpt_is_small)

    def run(self):
        # iterations
        while self._count < self._iterations:
            self.get_data()
            self.parse_data()
            self.train_video()
            self.log()
            self.save()
            self._count += 1

    def train_video(self):
        self._frame_id = 0
        while True:
            if self.get_frame() is None:
                break
            self.preprocess()
            self.train()
            self.postprocess()
            self._frame_id += 1

    def parse_data(self):

        for db_name, database in self._datas.items():
            if 'database' in database and isinstance(database['database'], (list, tuple)):
                database['database'] = database['database'][0]
            if "type" in database and isinstance(database['type'], (list, tuple)):
                database['type'] = database['type'][0]
            if "data_type" in database and isinstance(database['data_type'], (list, tuple)):
                database['data_type'] = database['data_type'][0]
            if 'frame_num' in database:
                database['frame_num'] = database['frame_num'][0].item()

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
        mask = []
        no_mask = True
        for db_name, database in self._datas.items():
            batch_size, frames, h, w = database['sensor_raw'].shape
            max_h = max_h if max_h > h else h
            max_w = max_w if max_w > w else w
            if self._frame_id < frames:
                sensor_raw.append(database['sensor_raw'][:, self._frame_id].unsqueeze(dim=1))
                gt.append(database['gt'][:, self._frame_id].unsqueeze(dim=1))

                if 'weight_mask' in database:
                    no_mask = False
                    mask.append(database['weight_mask'][:, self._frame_id].unsqueeze(dim=1))

                else:
                    mask.append(None)

                read.append(database["cam"]["read"])
                shot.append(database["cam"]["shot"])
                black_level.append(database["cam"]['black_level'])
                maximum.append(database["cam"]['maximum'])
                frame_info.append({"database": db_name,
                                   "frame_id": self._frame_id,
                                   "frame_num": batch_size})

        if len(sensor_raw) > 0:
            for data_id in range(len(sensor_raw)):
                sensor_raw[data_id] = shape_align(sensor_raw[data_id], max_h, max_w)
                gt[data_id] = shape_align(gt[data_id], max_h, max_w)
                if not no_mask:
                    if mask[data_id] is None:
                        mask[data_id] = torch.zeros((gt[data_id].shape[0], 1, max_h, max_w), dtype=torch.float32)
                    else:
                        mask[data_id] = shape_align(mask[data_id], max_h, max_w)
            if no_mask:
                mask = None
            this_batch_frames = {"sensor_raw": torch.concat(sensor_raw, dim=0).to(self.device),
                                 "clean_raw": torch.concat(gt, dim=0).to(self.device),
                                 "weight_mask": torch.concat(mask, dim=0).to(self.device) if mask is not None else None,
                                 "read": torch.concat(read).to(self.device),
                                 "shot": torch.concat(shot).to(self.device),
                                 'black_level': torch.concat(black_level).to(self.device),
                                 "maximum": torch.concat(maximum).to(self.device),
                                 "frame_info": frame_info}
            if self._frame_sequence is None:
                self._frame_sequence = {}
            self._frame_sequence[self._frame_id] = this_batch_frames
            return this_batch_frames
        else:
            return None

    def prepare_losses(self):

        losses = None

        for name, loss in self._losses.items():
            if not (loss["start_iter"] <= self._count <= loss["end_iter"]):
                continue
            if losses is None:
                losses = {}
            if ((name in ("l1", "pixel_l1", "channel_l1", "mse", "smooth_l1")) and
                    loss['init_parameters'].get("denoise", False) and
                    loss['init_parameters']['denoise'].get("enable", False)):
                losses["denoise_{}".format(name)] = deepcopy(loss)
                losses["denoise_{}".format(name)]['weight'] *= loss['init_parameters']['denoise'].get("weight", 1.0)
                losses["denoise_{}".format(name)]["input"] = [self._outputs["denoise"], self._gt]
                losses["fusion_{}".format(name)] = deepcopy(loss)
                losses["fusion_{}".format(name)]["input"] = [self._outputs["fusion"], self._gt]
            elif name in ("masked_mse", "masked_l1"):
                mask = self._frame_sequence[self._frame_id].get("weight_mask")
                assert mask is not None
                if loss['init_parameters'].get("denoise", False) and loss['init_parameters']['denoise'].get("enable", False):
                    losses["denoise_{}".format(name)] = deepcopy(loss)
                    losses["denoise_{}".format(name)]['weight'] *= loss['init_parameters']['denoise'].get("weight", 1.0)
                    losses["denoise_{}".format(name)]["input"] = [self._outputs["denoise"], self._gt, mask]
                    losses["fusion_{}".format(name)] = deepcopy(loss)
                    losses["fusion_{}".format(name)]["input"] = [self._outputs["fusion"], self._gt, mask]
                else:
                    losses[name] = deepcopy(loss)
                    losses[name]["input"] = [self._outputs["fusion"], self._gt, mask]
            elif name in ("gradient"):

                losses[name] = deepcopy(loss)
                losses[name]["input"] = [torch.nn.functional.pixel_unshuffle(self._outputs["fusion"], 2),
                                         torch.nn.functional.pixel_unshuffle(self._gt, 2)]
            else:
                losses[name] = deepcopy(loss)
                losses[name]["input"] = [self._outputs["fusion"], self._gt]

        assert losses is not None
        return losses

    def log(self):

        if self._count % self._log_interval and self._count != self._iterations:
            return

        self._log.info("-" * 100)
        self._log.info("{}{:^20}{}".format("-" * 40, "ITERATION:{:08d}".format(self._count), "-" * 40))
        self._log.info("-" * 100)

        # title
        head = "{:<8}".format("Frame")
        for name, loss in self._results[0]["losses"].items():
            if name.lower() == "total":
                continue
            head = "{}|{:<20}{:<10}".format(head, name, "weight")
            if "components" in loss:
                for sub_name in loss["components"].keys():
                    if "total" in sub_name:
                        continue
                    else:
                        head = "{}{:<20}{:<10}".format(head, sub_name, "weight")
        head = "{}|{:<20}".format(head, "total")
        self._log.info(head)

        # losses
        for frame_id, frame_result in self._results.items():
            head = "{:<8}".format(frame_id)
            total_loss = None
            for name, loss in frame_result["losses"].items():
                if name.lower() == "total":
                    total_loss = loss
                    continue
                head = "{}|{:<20}{:<10}".format(head,
                                                "{:.8f}".format(loss["value"].item()),
                                                "{:.4f}".format(loss["weight"]))
                if "components" in loss:
                    for sub_name, sub_loss in loss["components"].items():
                        if "total" in sub_name:
                            continue
                        else:
                            head = "{}{:<20}{:<10}".format(head,
                                                           "{:.8f}".format(sub_loss.item()
                                                                           if isinstance(sub_loss, torch.Tensor)
                                                                           else (sub_loss["value"].item()
                                                                                 if isinstance(sub_loss["value"],
                                                                                               torch.Tensor)
                                                                                 else sub_loss["value"])),
                                                           "{:.4f}".format(sub_loss.get("weight")))
            head = "{}|{:<10}".format(head, "{:.8f}".format(total_loss.item()))
            self._log.info(head)

        # add loss
        metrics = {}
        last_result = next(reversed(self._results.items()), None)
        for name, loss in last_result[-1]["losses"].items():
            if name.lower() == "total":
                metrics["train/{}".format(name)] = loss.item()
            else:
                metrics["train/losses/{}".format(name)] = loss["value"].item()
                if "components" in loss:
                    for sub_name, sub_loss in loss["components"].items():
                        if "total" in sub_name:
                            continue
                        else:
                            metrics["train/losses/{}/{}".format(name, sub_name)] = (sub_loss.item()
                                                                             if isinstance(sub_loss, torch.Tensor)
                                                                             else (sub_loss["value"].item()
                                                                                   if isinstance(sub_loss["value"],
                                                                                                 torch.Tensor)
                                                                                   else sub_loss["value"]))
        self._log.log_metrics(metrics, self._count)

    def evaluate(self):
        if self._evaluator is None:  # or (self._count % self._save_interval and self._count != self._iterations):
            return
        self._evaluator.set_training_iter(self._count)
        self._eval_result = self._evaluator()
        # add evaluate metrics to log
        metrics = {}

        return self.get_model_eval_parameters()

    def get_model_eval_parameters(self):
        eval_score = 0
        for database_name, database_metrics in self._eval_result.items():
            eval_score += database_metrics[self._evaluator._dominate_metrics]['mean']

        if self._evaluator._dominate_metrics in ["psnr"]:
            best_ckpt_is_small = False
        else:
            best_ckpt_is_small = True
        return eval_score/len(self._eval_result), best_ckpt_is_small

    def map_result_to_data(self):

        for sequence_id, frames in self._frame_sequence.items():
            frame_info = frames['frame_info']
            start_id = 0
            for database in frame_info:
                db_name = database['database']
                end_id = start_id + database['frame_num']
                if "sensor_raw_vst" not in self._datas[db_name]:
                    self._datas[db_name]["sensor_raw_vst"] = []
                self._datas[db_name]["sensor_raw_vst"].append(frames['net_inputs']['present'][start_id:end_id])
                if "gt_vst" not in self._datas[db_name]:
                    self._datas[db_name]["gt_vst"] = []
                self._datas[db_name]["gt_vst"].append(frames['gt'][start_id:end_id])
                if "denoise_vst" not in self._datas[db_name]:
                    self._datas[db_name]["denoise_vst"] = []
                self._datas[db_name]["denoise_vst"].append(frames['net_outputs']['denoise'][start_id:end_id])
                if "fusion_vst" not in self._datas[db_name]:
                    self._datas[db_name]["fusion_vst"] = []
                self._datas[db_name]["fusion_vst"].append(frames['net_outputs']['fusion'][start_id:end_id])
                if "noise" not in self._datas[db_name]:
                    self._datas[db_name]["noise"] = []
                self._datas[db_name]["noise"].append(frames['net_outputs']['noise'][start_id:end_id])
                if "fusion_mask" not in self._datas[db_name]:
                    self._datas[db_name]["fusion_mask"] = []
                self._datas[db_name]["fusion_mask"].append(frames['net_outputs']['fusion_mask'][start_id:end_id])
                if "sensor_raw_denoise" not in self._datas[db_name]:
                    self._datas[db_name]["sensor_raw_denoise"] = []
                self._datas[db_name]["sensor_raw_denoise"].append(frames['sensor_raw_denoise'][start_id:end_id])
                if "sensor_raw_fusion" not in self._datas[db_name]:
                    self._datas[db_name]["sensor_raw_fusion"] = []
                self._datas[db_name]["sensor_raw_fusion"].append(frames['sensor_raw_fusion'][start_id:end_id])
                start_id = end_id

        target_names = ["sensor_raw_vst", "gt_vst", "denoise_vst", "fusion_vst", "noise", "fusion_mask",
                        "sensor_raw_denoise", "sensor_raw_fusion"]
        for db_name, data in self._datas.items():
            for name in target_names:
                if name in data:
                    data[name] = torch.concat(data[name], dim=1)

    def update_best_ckpt(self, model_val, model_path, best_ckpt_is_small=True):
        rm_ckpt = self._best_ckpts.insert({self._best_ckpts.sort_key: -model_val if best_ckpt_is_small else model_val,
                                           "path": model_path})
        if rm_ckpt is not None:
            self._log.info("Remove CKPT: {}\t{}".format(rm_ckpt["path"], rm_ckpt["score"]))
        self._log.info("Reserved CKPTs: ")
        ckpts = self._best_ckpts.get_sorted_items()
        for ckpt_id, ckpt in enumerate(ckpts):
            self._log.info("[{}/{}]\t{}\t{}".format(ckpt_id,
                                                    self._reserve_best_ckpt,
                                                    ckpt["path"], ckpt["score"]))

    def save(self):

        if self._count % self._save_interval and self._count != self._iterations:
            return

        # vis the training process
        if self._vis is None or not self._vis.get("enable", False):
            return

        # to partially vis samples
        vis_sample_num = self._vis.get("sample_per_db", 2)
        vis_frames = self._vis.get("frames", 3)
        image_save_dir =  os.path.join(self._save_path, "image")
        self._log.info("-" * 100)
        self._log.info("{}".format("ITERATION:{:08d} SAVE TRAINING DATA TO: {}".format(self._count,
                                                                                       image_save_dir)))
        self._log.info("-" * 100)

        # vis net inputs and outputs
        self.map_result_to_data()
        target_names = ["noise", "fusion_mask", "weight_mask"]  # ["sensor_raw_vst", "gt_vst", "denoise_vst", "fusion_vst", "noise", "mask"]
        cvt_rgb_names = ["sensor_raw", "gt", "sensor_raw_denoise", "sensor_raw_fusion"]
        for db_name, data in self._datas.items():
            for sample_id, sample_name in enumerate(data['sample_name']):
                if sample_id >= vis_sample_num:
                    break
                key = "train/ITER:{:08d}/S:{:03d}/{}/AGain:{}/DGain:{}".format(self._count, sample_id, sample_name,
                                                                    data['cam']['analog_gain'][sample_id].to(torch.int32),
                                                                    data['cam']['d_gain'][sample_id].round().to(torch.int32))
                image_names = [os.path.basename(batch_files[sample_id]).split(".")[0] for batch_files in data["files"]]
                for target_id, target_name in enumerate(target_names):
                    target_key = "{}/{}_{}".format(key, target_id, target_name)
                    sample = data[target_name][sample_id]
                    if target_name == "fusion_mask":
                        sample_norm = (sample * 255).round().to(torch.uint8)
                        sample_list = list(sample_norm.detach().cpu().numpy())
                    elif target_name == "weight_mask":
                        sample_norm = (sample / sample.max() * 255).round().to(torch.uint8)
                        sample_list = list(sample_norm.detach().cpu().numpy())
                    else:
                        mean_val, std_val = sample.mean(), sample.std()
                        clip_min = max(mean_val - 3 * std_val, 0)
                        clip_max = mean_val + 3 * std_val
                        sample_norm = normalization(sample, clip_min, clip_max, 0, 255).round().to(
                            torch.uint8).unsqueeze(
                            dim=1)
                        sample_list = list(torch.nn.PixelUnshuffle(2)(sample_norm).detach().cpu().numpy())
                        sample_list = [np.vstack((np.hstack((sample[0], sample[1])),
                                                  np.hstack((sample[2], sample[3])))) for sample in sample_list]
                    end_idx = min(len(image_names), vis_frames)
                    sample_dict = dict(zip(image_names[:end_idx], sample_list[:end_idx]))
                    # save images to trackio .db
                    if self._vis.get("save2db", False):
                        self._log.log_image_sequence(sample_dict, key=target_key)
                    # save to local folder
                    if self._vis.get("save2local", False):
                        self.save_images(sample_dict, image_save_dir,
                                         target_key.replace(":", "-").replace("/", "_"))
                for target_id, target_name in enumerate(cvt_rgb_names):
                    target_key = "{}/{}_{}".format(key, target_id + len(target_names), target_name)
                    sample = data[target_name][sample_id]
                    frame_names = []
                    sample_list = []
                    for frame_id, raw in enumerate(sample):
                        bgr = raw2bgr(raw.detach().cpu().numpy(),
                                      data['cam']['bits'][sample_id].item(),
                                      data['cam']['bayer_pattern'][sample_id],
                                      data['cam']['black_level'][sample_id].item(),
                                      (data['cam']['rgb_gain'][sample_id] * data['cam']['r_gain'][sample_id] * data['cam']['d_gain'][sample_id]).item(),
                                      (data['cam']['rgb_gain'][sample_id] * data['cam']['g_gain'][sample_id] * data['cam']['d_gain'][sample_id]).item(),
                                      (data['cam']['rgb_gain'][sample_id] * data['cam']['b_gain'][sample_id] * data['cam']['d_gain'][sample_id]).item(),
                                      ccm=data['cam']['ccm'][sample_id].cpu().numpy())
                        frame_names.append(
                            "seq{}_{}".format(frame_id, image_names[data['frame_id'][sample_id, frame_id]]))
                        sample_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    end_idx = min(len(image_names), vis_frames)
                    sample_dict = dict(zip(frame_names[:end_idx], sample_list[:end_idx]))
                    # save images to trackio .db
                    if self._vis.get("save2db", False):
                        self._log.log_image_sequence(sample_dict, key=target_key)
                    # save to local folder
                    if self._vis.get("save2local", False):
                        self.save_images(sample_dict, image_save_dir,
                                         target_key.replace(":", "-").replace("/", "_"),
                                         rgb2bgr=True)
                    if "weight_mask" in data:
                        target_key = "{}/{}_{}".format(key, target_id + len(target_names) + len(cvt_rgb_names), target_name)
                        masked_sample = {}
                        for frame_id, (name, rgb) in enumerate(sample_dict.items()):
                            mask = data["weight_mask"][sample_id][frame_id]
                            name = "{}_masked".format(name)
                            masked_sample[name] = rgb
                            masked_sample[name][mask==0] = 0
                        if self._vis.get("save2db", False):
                            self._log.log_image_sequence(masked_sample, key=target_key)
                        # save to local folder
                        if self._vis.get("save2local", False):
                            self.save_images(masked_sample, image_save_dir,
                                             target_key.replace(":", "-").replace("/", "_"),
                                             rgb2bgr=True)

        # Do not save the model of 1st iter
        if self._count == 0:  #  and self._evaluate_before_train:
            return

        # save ckpt
        self._log.info("-" * 100)
        self._log.info("{}{:^40}{}".format("-" * 30, "ITERATION:{:08d} SAVE MODEL".format(self._count), "-" * 30))
        self._log.info("-" * 100)
        model_path = os.path.join(self._save_path, "ckpt",
                                  "{}_{:08d}.pth".format(self._model.__class__.__name__,
                                                         self._count))
        self.save_model(model_path)
        self._log.info("Save PTH: {}".format(model_path))

        # update best ckpts
        if self._evaluator is None:
            last_result = next(reversed(self._results.items()), None)
            model_val = last_result[-1]["losses"]["total"].item()
            best_ckpt_is_small = True
        else:
            model_val, best_ckpt_is_small = self.evaluate()
        self.update_best_ckpt(model_val, model_path, best_ckpt_is_small=best_ckpt_is_small)

    def finish(self):
        self.log()
        self.save()
        self._log.info("Training pipeline finished. Saving final model ...")
        ckpts = self._best_ckpts.get_sorted_items()
        for ckpt_id, ckpt in enumerate(ckpts):
            self._log.info("[{}/{}]\t{}\t{}".format(ckpt_id,
                                                    self._reserve_best_ckpt,
                                                    ckpt["path"], ckpt["score"]))
        super().finish()

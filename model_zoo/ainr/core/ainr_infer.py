import os

import numpy as np
import torch
import cv2

from aifactory.core import BasicInference
from aifactory.libs.nn.functionals import vst, ivst
from aifactory.libs.common.raw2rgb import raw2bgr


class AinrInference(BasicInference):
    _frame_id = 0
    _frame_sequence = None

    def __init__(self, model, ckpt=None, device=torch.device("cpu"), log=None, **kwargs):
        super().__init__(model, ckpt, device=device, log=log, **kwargs)

    # def __call__(self, *args, **kwargs):
    #     if kwargs.get('work_mode')=="deploy":
    #         return self.deploy(*args, **kwargs)
    #     else:
    #         self._log.error("Unrecognized work mode. Stop inference.")
    #         return None

    def deploy(self, *args, **kwargs):
        self.run_oneshot(*args, **kwargs)
        result = self._results[self._frame_id]
        self._frame_id += 1
        return result

    def export_onnx(self):
        self._log.error("export onnx of {} is not ready".format(self.__class__.__name__))

    def run(self,  *args, **kwargs):
        with torch.no_grad():
            if kwargs.get('work_mode')=="deploy":
                return self.deploy(*args, **kwargs)
            else:
                self._log.error("Unrecognized work mode. Stop inference.")
                return None

    def run_oneshot(self, *args, **kwargs):
        self.parse_data(*args, **kwargs)
        self.preprocess()
        self.model_infer()
        self.postprocess(enable_raw2bgr=True)
        return

    def parse_data(self, *args, **kwargs):
        # input raw and meta for inference
        if len(args) == 2:
            if self._frame_sequence is None:
                self._frame_sequence = {}
            self._frame_sequence[self._frame_id] = {
                "sensor_raw": torch.from_numpy(args[0]).to(device=self._device, dtype=self._dtype).reshape(
                    1, 1, args[1]['height'], args[1]['width'])}
            for key, val in args[1].items():
                if isinstance(val, str):
                    val = [val]
                elif isinstance(val, (int, float)):
                    val = torch.tensor(val).to(self.device).unsqueeze(dim=0)
                elif isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).to(self.device).unsqueeze(dim=0)
                else:
                    # raise ValueError("Do not support camera parameter type of {}".format(type(val)))
                    pass
                self._frame_sequence[self._frame_id][key] = val

        if self._datas is None:
            return
        for db_name, database in self._datas.items():
            if 'database' in database and isinstance(database['database'], (list, tuple)):
                database['database'] = database['database'][0]
            if "type" in database and isinstance(database['type'], (list, tuple)):
                database['type'] = database['type'][0]
            if "data_type" in database and isinstance(database['data_type'], (list, tuple)):
                database['data_type'] = database['data_type'][0]
            if 'frame_num' in database:
                database['frame_num'] = database['frame_num'][0].item()

    def preprocess(self):

        # process inputs
        this_batch_frames = self._frame_sequence[self._frame_id]
        sample_num = this_batch_frames["sensor_raw"].shape[0]
        vst_samples = []
        vst_gts = []
        for sample_id in range(sample_num):
            vst_samples.append(
                vst(this_batch_frames["sensor_raw"][sample_id].to(dtype=torch.float32, device=self._device) -
                    this_batch_frames["black_level"][sample_id].to(dtype=torch.float32, device=self._device),
                    this_batch_frames["shot"][sample_id],
                    this_batch_frames["read"][sample_id]).unsqueeze(dim=0))

            if "clean_raw" in this_batch_frames:
                vst_gts.append(
                    vst(this_batch_frames["clean_raw"][sample_id].to(dtype=torch.float32, device=self._device) -
                        this_batch_frames["black_level"][sample_id].to(dtype=torch.float32, device=self._device),
                        this_batch_frames["shot"][sample_id],
                        this_batch_frames["read"][sample_id]).unsqueeze(dim=0))
        # prepare net inputs
        present = torch.concat(vst_samples, dim=0)
        if self._frame_id == 0:
            previous = torch.zeros_like(present)
        else:
            previous = self._results[self._frame_id - 1]["feedback"]
        self._inputs = {"present": present.to(dtype=self.dtype, device=self.device),
                        "previous": previous.to(dtype=self.dtype, device=self.device)}
        this_batch_frames["net_inputs"] = self._inputs

        if len(vst_gts) > 0:
            this_batch_frames["gt"] = torch.concat(vst_gts, dim=0).to(dtype=self.dtype, device=self.device)
            self._gt = this_batch_frames["gt"]
        return self._inputs

    def map_sample_to_db(self, sample_id):
        this_batch_frames = self._frame_sequence[self._frame_id]
        start_id = 0
        for frame_info in this_batch_frames['frame_info']:
            end_id = start_id + frame_info['frame_num']
            if end_id > sample_id >= start_id:
                return frame_info['database'], sample_id - start_id
            else:
                start_id = end_id
        return None

    def get_cam_param_from_database(self, sample_id, param_name):
        sample_db_name, sample_db_idx = self.map_sample_to_db(sample_id)
        data = self._datas[sample_db_name]
        param = data['cam'][param_name][sample_db_idx]
        if not isinstance(param, torch.Tensor):
            return param
        elif param.numel() == 1:
            return param.item()
        else:
            return param.cpu().numpy()

    def postprocess(self, enable_raw2bgr=False):
        # record result
        if self._results is None:
            self._results = {self._frame_id: {}}
        elif self._frame_id not in self._results:
            self._results[self._frame_id] = {}
        else:
            pass
        self._results[self._frame_id]["feedback"] = self._outputs["fusion"].detach()

        # process data from net outputs
        this_batch_frames = self._frame_sequence[self._frame_id]
        this_batch_frames["net_outputs"] = self._outputs
        sample_num = this_batch_frames["sensor_raw"].shape[0]
        raw_denoise = []
        raw_fusion = []

        # convert raw to bgr, always for inference and evaluation
        if enable_raw2bgr:
            bgr_raw = []
            bgr_fusion = []
            bgr_gt = []
            clear_raw = []
        else:
            bgr_raw = None
            bgr_fusion = None
            bgr_gt = None

        for sample_id in range(sample_num):
            raw_denoise.append(torch.clip(ivst(self._outputs["denoise"][sample_id],
                                               this_batch_frames["shot"][sample_id],
                                               this_batch_frames["read"][sample_id]).unsqueeze(dim=0) +
                                          this_batch_frames["black_level"][sample_id].to(dtype=torch.float32,
                                                                                         device=self._device),
                                          0, this_batch_frames["maximum"][sample_id]))
            raw_fusion.append(torch.clip(ivst(self._outputs["fusion"][sample_id],
                                              this_batch_frames["shot"][sample_id],
                                              this_batch_frames["read"][sample_id]).unsqueeze(dim=0) +
                                         this_batch_frames["black_level"][sample_id].to(dtype=torch.float32,
                                                                                        device=self._device),
                                         0, this_batch_frames["maximum"][sample_id]))
            if enable_raw2bgr:
                # get raw2bgr parameters
                raw2bgr_keys = ['bits', 'bayer_pattern', 'black_level', "d_gain", "drc_gain", 'r_gain', 'g_gain',
                                'b_gain', 'ccm']
                raw2bgr_params = {}
                for key in raw2bgr_keys:
                    if key in this_batch_frames:
                        p = this_batch_frames[key][sample_id]
                        if isinstance(p, torch.Tensor):
                            p = p.item() if p.numel() == 1 else p.cpu().numpy()
                        raw2bgr_params[key] = p
                    else:
                        raw2bgr_params[key] = self.get_cam_param_from_database(sample_id, key)

                bgr = raw2bgr(this_batch_frames["sensor_raw"][sample_id].squeeze().detach().cpu().numpy(),
                              raw2bgr_params['bits'],
                              raw2bgr_params['bayer_pattern'],
                              raw2bgr_params['black_level'],
                              raw2bgr_params['r_gain'] * raw2bgr_params['d_gain'] * raw2bgr_params['drc_gain'],
                              raw2bgr_params['g_gain'] * raw2bgr_params['d_gain'] * raw2bgr_params['drc_gain'],
                              raw2bgr_params['b_gain'] * raw2bgr_params['d_gain'] * raw2bgr_params['drc_gain'],
                              ccm=raw2bgr_params['ccm'])
                bgr_raw.append(bgr)
                # bgr_raw.append(torch.from_numpy(bgr).unsqueeze(dim=0).permute(0, 3, 1, 2).to(self.device))
                bgr = raw2bgr(raw_fusion[-1].squeeze().detach().cpu().numpy(),
                              raw2bgr_params['bits'],
                              raw2bgr_params['bayer_pattern'],
                              raw2bgr_params['black_level'],
                              raw2bgr_params['r_gain'] * raw2bgr_params['d_gain'] * raw2bgr_params['drc_gain'],
                              raw2bgr_params['g_gain'] * raw2bgr_params['d_gain'] * raw2bgr_params['drc_gain'],
                              raw2bgr_params['b_gain'] * raw2bgr_params['d_gain'] * raw2bgr_params['drc_gain'],
                              ccm=raw2bgr_params['ccm'])
                bgr_fusion.append(bgr)
                # bgr_fusion.append(torch.from_numpy(bgr).unsqueeze(dim=0).permute(0, 3, 1, 2).to(self.device))
                if self._datas is not None:
                    sample_db_name, sample_db_idx = self.map_sample_to_db(sample_id)
                    data = self._datas[sample_db_name]
                    bgr = raw2bgr(data['gt'][sample_id][self._frame_id].squeeze().detach().cpu().numpy(),
                                  raw2bgr_params['bits'],
                                  raw2bgr_params['bayer_pattern'],
                                  raw2bgr_params['black_level'],
                                  raw2bgr_params['r_gain'] * raw2bgr_params['d_gain'] * raw2bgr_params['drc_gain'],
                                  raw2bgr_params['g_gain'] * raw2bgr_params['d_gain'] * raw2bgr_params['drc_gain'],
                                  raw2bgr_params['b_gain'] * raw2bgr_params['d_gain'] * raw2bgr_params['drc_gain'],
                                  ccm=raw2bgr_params['ccm'])
                    bgr_gt.append(bgr)
                    # bgr_gt.append(torch.from_numpy(bgr).unsqueeze(dim=0).permute(0, 3, 1, 2).to(self.device))
                else:
                    bgr_gt = None

        this_batch_frames["sensor_raw_denoise"] = torch.concat(raw_denoise, dim=0)
        this_batch_frames["sensor_raw_fusion"] = torch.concat(raw_fusion, dim=0)
        self._results[self._frame_id]["sensor_raw_denoise"] = this_batch_frames["sensor_raw_denoise"].round().to(torch.uint16)
        self._results[self._frame_id]["sensor_raw_fusion"] = this_batch_frames["sensor_raw_fusion"].round().to(torch.uint16)
        if enable_raw2bgr:
            self._results[self._frame_id]["srgb"] = {"raw": bgr_raw,
                                                     "fusion": bgr_fusion}
            this_batch_frames["bgr_raw"] = torch.from_numpy(np.array(bgr_raw)).permute(0,3,1,2) if bgr_raw is not None else None
            this_batch_frames["bgr_fusion"] = torch.from_numpy(np.array(bgr_fusion)).permute(0,3,1,2) if bgr_fusion is not None else None
            this_batch_frames["bgr_gt"] = torch.from_numpy(np.array(bgr_gt)).permute(0,3,1,2) if bgr_gt is not None else None

        # the following is for training process
        if "losses" in self._results:
            losses = self._results.pop("losses")
            if self._frame_id in self._results:
                self._results[self._frame_id]["losses"] = losses
            else:
                self._results[self._frame_id] = {"losses": losses}

    def save_images(self, images, save_dir, key=None, rgb2bgr=False):
        os.makedirs(save_dir, exist_ok=True)
        for name, image in images.items():
            if key is None:
                file_name = os.path.join(save_dir, "{}.png".format(key, name))
            else:
                file_name = os.path.join(save_dir, "{}_{}.png".format(key, name))
            if rgb2bgr:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def reset(self):
        self._frame_id = 0
        self._frame_sequence = None

        self._datas = None
        self._results = None
        self._inputs = None
        self._outputs = None

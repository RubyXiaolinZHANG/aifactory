import sys
sys.path.append("../../../../")
import os.path
from shutil import copyfile
from copy import deepcopy
import cv2
import numpy as np
import torch
from aifactory.utils.load_file import load_file
from aifactory.libs.common.bgr2raw import bgr2raw
from aifactory.libs.common.raw2rgb import raw2bgr
from aifactory.libs.common.image_processor import enhanced_edge_detection
from aifactory.utils.save_files import save_as_image
from aifactory.utils.seed import set_seed
# from raw_augment import add_noise_to_raw
# from camera import IspParameters, CAMERAS
from .raw_augment import add_noise_to_raw
from .camera import IspParameters, CAMERAS


class DatasetVimeo2Raw(torch.utils.data.Dataset):
    _dataset = None
    _cam = None
    _process_frame_num = None
    _iso = None
    _fix_sequence = False
    _log = None
    _dst_bits = None
    _crop_area = None
    _edge_mask_enable = None
    _edge_mask_val = None

    def __init__(self, paths, cam, cvt_bits=None, crop=None, frame_num=7, iso=None, edge_mask_val=None,
                 record_bad_data=None, fix_sequence=False,
                 log=None):
        if isinstance(paths, str):
            assert os.path.exists(paths)
            self._dataset = load_file(paths)
        else:
            for path in paths:
                assert os.path.exists(path)
                if self._dataset is None:
                    self._dataset = {}
                self._dataset.update(load_file(path))
        if isinstance(cam, IspParameters):
            self._cam = cam
        else:
            self._cam = CAMERAS[cam]()
        self._process_frame_num = frame_num
        self._iso = iso

        self._edge_mask_val = edge_mask_val
        self._edge_mask_enable = edge_mask_val is not None

        self._samples = list(self._dataset.keys())
        self._item = None
        if log is not None:
            self._log = log
        else:
            self._log = None

        if record_bad_data is not None:
            os.makedirs(os.path.dirname(record_bad_data), exist_ok=True)
            self._log_file = open(record_bad_data, "w")
        else:
            self._log_file = None
        self._fix_sequence = fix_sequence
        self._dst_bits = cvt_bits if cvt_bits is not None else self._cam.bits

        # crop setting
        if isinstance(crop, (tuple, list)):
            self._crop_area={"height": crop[0],
                             "width": crop[1]}
        elif isinstance(crop, dict):
            self._crop_area={"height": crop["height"],
                             "width": crop["width"]}
        elif crop is None:
            pass
        else:
            raise ValueError("do not support crop parameter of type: {}".format(type(crop)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):

        self._item = item
        sample = deepcopy(self._dataset[self._samples[item]])

        try:
            frame_ids = self.get_sequence(sample)
            self._cam.get_tuning_parameters(self._iso)
            # self._cam.get_tuning_parameters(100)
            crop_roi = None
            frames = {"sensor_raw": [],
                      "gt": [],
                      "frame_id": []}
            if self._edge_mask_enable:
                frames["weight_mask"] = []

            for frame_id in frame_ids:
                bgr = cv2.imread(sample['files'][frame_id])
                assert bgr is not None, "{} is none".format(sample['files'][frame_id])
                if self.crop_enable:
                    if crop_roi is None:
                        im_height, im_width, _ = bgr.shape
                        sc = np.random.randint(0, im_width-self.crop_w + 1)
                        sr = np.random.randint(0, im_height-self.crop_h + 1)
                        crop_roi = {"x1": sc,
                                    "x2": sc + self.crop_w,
                                    "y1": sr,
                                    "y2": sr + self.crop_h}
                    bgr = bgr[crop_roi["y1"]: crop_roi["y2"], crop_roi["x1"]: crop_roi["x2"]]

                if self._edge_mask_enable:
                    edge = enhanced_edge_detection(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                                                   threshold=12, min_area=20, min_size=10,
                                                   remove_boundary=0, edge_val=self._edge_mask_val)
                    frames["weight_mask"].append(edge)

                # bgr to raw
                raw_norm = bgr2raw(bgr, self._cam, return_normalized_bgr=True)
                raw = np.round(raw_norm * ((1 << self._dst_bits) - 1))
                sensor_raw = self.add_noise(raw)
                frames["sensor_raw"].append(
                    sensor_raw.astype(np.uint16) if self._cam.bits <= 16 else sensor_raw.astype(np.uint32))
                frames["gt"].append(raw.astype(np.uint16) if self._cam.bits <= 16 else raw.astype(np.uint32))
                frames["frame_id"].append(frame_id)

            frames["sensor_raw"] = np.array(frames["sensor_raw"])
            frames["gt"] = np.array(frames["gt"])
            frames["frame_id"] = np.array(frames["frame_id"])
            if self._edge_mask_enable:
                frames["weight_mask"] = np.array(frames["weight_mask"])

            sample.update({"sample_name": self._samples[item],
                           "cam": self.get_cam_params(),
                           "sequence_length": self._process_frame_num
                           })
            sample.update(frames)

        except:
            bad_samples = []
            for file in sample['files']:
                bgr = cv2.imread(file)
                if bgr is None:
                    bad_samples.append(file)
            self.record(bad_samples)
            item2 = np.random.randint(len(self._dataset))
            info = "replace the damage sample {} with {}".format(self._samples[item],
                                                                 self._samples[item2])
            if self._log is not None:
                self._log.warning(info)
            else:
                print(info)
            sample = self.__getitem__(item2)
            info = "{}{:^40}{}".format("*" * 30,
                                       "replacement succeeds!",
                                       "*" * 30)
            if self._log is not None:
                self._log.warning(info)
            else:
                print(info)
        return sample

    @property
    def crop_enable(self):
        if self._crop_area is None:
            return False
        else:
            return True

    @property
    def crop_h(self):
        return self._crop_area["height"] if self._crop_area is not None else None

    @property
    def crop_w(self):
        return self._crop_area["width"] if self._crop_area is not None else None

    def get_sequence(self, sample):
        frame_num = sample['frame_num']
        if self._fix_sequence:
            start_id = 0
        else:
            start_id = np.random.randint(frame_num)
        frame_sequence = [start_id]
        shift = 1
        while len(frame_sequence) < self._process_frame_num:
            if frame_sequence[-1] == frame_num - 1:
                shift = -1
            elif frame_sequence[-1] == 0:
                shift = 1
            frame_sequence.append(frame_sequence[-1] + shift)
        return frame_sequence

    def add_noise_org(self, raw):
        bit_shift = self._dst_bits - self._cam.bits
        black_level = self._cam.black_level * (1 << bit_shift)
        read = self._cam.read  * (1 << bit_shift * 2)
        shot = self._cam.shot * (1 << bit_shift)
        # sensor_raw = add_noise_to_raw(raw.astype(np.float32) - self._cam.black_level,
        #                               self._cam.bits, self._cam.read, self._cam.shot)
        # return np.clip(sensor_raw["sensor_raw"] + self._cam.black_level, 0, self._cam.maximum).astype(np.uint16)
        sensor_raw = add_noise_to_raw(raw.astype(np.float32) - black_level, self._dst_bits, read, shot)
        return np.clip(sensor_raw["sensor_raw"] + black_level, 0, (1 << self._dst_bits) - 1).astype(np.uint16)


    def add_noise(self, raw):
        bit_shift = self._dst_bits - self._cam.bits
        read = self._cam.read  * (1 << bit_shift * 2)
        shot = self._cam.shot * (1 << bit_shift)
        sensor_raw = add_noise_to_raw(raw.astype(np.float32), self._dst_bits, read, shot)
        return np.clip(sensor_raw["sensor_raw"], 0, (1 << self._dst_bits) - 1).astype(np.uint16)

    def get_cam_params(self):
        cam = self._cam.get_parameter_dict()
        bit_shift = self._dst_bits - cam["bits"]
        if bit_shift != 0:
            cam["bits"] = self._dst_bits
            scale = (1 << bit_shift)
            cam['black_level'] = cam['black_level'] * scale
            cam['read'] *= scale ** 2
            cam['shot'] *= scale
            cam['maximum'] = (1 << self._dst_bits) - 1

        return cam


    @staticmethod
    def save_data(sample, save_dir):
        for key, batched_raw in zip(["gt", "sensor"], [sample['gt'], sample['sensor_raw']]):
            for frame_id, raw in enumerate(batched_raw):
                src_file = sample["files"][sample['frame_id'][frame_id]]
                sava_path = os.path.join(save_dir,
                                         "{}_{}_{}_{}.png".format(frame_id,
                                                                  sample["sample_name"],
                                                                  os.path.basename(src_file).replace(".png", ""),
                                                                  key
                                                                  )).replace("\\", "/")
                os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                norm_raw = raw.astype(np.float32) / sample['cam']['maximum']
                save_as_image(norm_raw, sava_path)
                sava_path = sava_path.replace("{}.png".format(key), "{}_Quarter.png".format(key))
                save_as_image(np.vstack([np.hstack([norm_raw[0::2, 0::2], norm_raw[0::2, 1::2]]),
                                         np.hstack([norm_raw[1::2, 0::2], norm_raw[1::2, 1::2]])]), sava_path)
                bgr = raw2bgr(raw, sample['cam']['bits'], sample['cam']['bayer_pattern'], sample['cam']['black_level'],
                              sample['cam']['r_gain'] * sample['cam']['d_gain'],
                              sample['cam']['g_gain'] * sample['cam']['d_gain'],
                              sample['cam']['b_gain'] * sample['cam']['d_gain'],
                              ccm=sample['cam']['ccm'])
                sava_path = sava_path.replace("{}_Quarter.png".format(key), "{}_RGB.png".format(key))
                cv2.imwrite(sava_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                save_path = sava_path.replace("{}_RGB.png".format(key), "{}_SRC.png".format(key))
                copyfile(src_file, save_path)
            # exit()

    def finish(self):
        if self._log_file is not None:
            self._log_file.close()

    def record(self, files):
        self._log.warning("")
        info = "{}{:^40}{}".format("*" * 30,
                                   "the following files area damaged",
                                   "*" * 30)
        if self._log is not None:
            self._log.warning(info)
        else:
            print(info)

        for file in files:
            if self._log is not None:
                self._log.warning(file)
            else:
                print(file)
            if self._log_file is not None:
                self._log_file.write(file + "\n")
        if self._log_file is not None:
            self._log_file.write("\n")

    def set_log(self, log):
        self._log = log

    def remove_log(self):
        self._log = None

    def set_bad_data_file(self, file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        self._log_file = open(file, "w")

    def remove_bad_data_file(self):
        self._log_file = None


def vimeo2raw_dataloader2samples(data):
    # sequence_length = len(data['frames'])
    batch_size = data['sensor_raw'].shape[0]
    samples = []
    sample_keys = list(data)
    for batch_id in range(batch_size):
        sample = {}
        for key in sample_keys:
            if key == "files":
                sample[key] = []
                for file in data['files']:
                    sample[key].append(file[batch_id])
            elif key == "cam":
                cam_params = {}
                for cam_key, cam_val in data[key].items():
                    if isinstance(cam_val, torch.Tensor):
                        cam_params[cam_key] = cam_val[batch_id].item() if cam_val[batch_id].numel == 1 else cam_val[
                            batch_id].cpu().detach().numpy()
                    else:
                        cam_params[cam_key] = cam_val[batch_id]
                sample[key] = cam_params
            elif key == "frames":
                sample[key] = []
                for batched_frame in data[key]:
                    frame = {}
                    for frame_key, frame_val in batched_frame.items():
                        if isinstance(frame_val, torch.Tensor):
                            frame[frame_key] = frame_val[batch_id].item() if frame_val[batch_id].numel == 1 else \
                                frame_val[batch_id].cpu().detach().numpy()
                        else:
                            frame[frame_key] = frame_val[batch_id]
            elif isinstance(data[key], torch.Tensor):
                sample[key] = data[key][batch_id].item() if data[key][batch_id].numel == 1 else data[key][
                    batch_id].cpu().detach().numpy()
            else:
                sample[key] = data[key][batch_id]
        samples.append(sample)
    return samples


def test_dataset_vimeo2raw():
    from tqdm import tqdm
    from camera import O2S

    # setting
    set_seed()
    batch_size = 16
    max_iters = 14
    test_sample_num = batch_size * max_iters

    dataset = DatasetVimeo2Raw("F:/database/vimeo_png/datasets/vimeo_val_sn-200.yaml",
                               O2S(), frame_num=5)
    save_dir = "H:/_result/ai_factory/raw_factory/DatasetVimeo2Raw"
    tqdm.write("\n{} test dataset {}\n save to:\t{}".format("*" * 50, "*" * 50, save_dir))
    with tqdm(total=test_sample_num, desc="test dataset") as pbar:
        for sample_id, sample in enumerate(dataset):
            save_data_to = os.path.join(save_dir, "S{:06d}_{}".format(sample_id, sample["sample_name"]))
            DatasetVimeo2Raw.save_data(sample, save_data_to)
            pbar.update(1)
            if sample_id == test_sample_num - 1:
                break

    # test dataloader
    save_dir = "H:/_result/ai_factory/raw_factory/DataLoaderVimeo2Raw"
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    data_iter = iter(dataloader)
    iters = 0

    tqdm.write("\n{} test dataloader {}\n save to:\t{}".format("*" * 50, "*" * 50, save_dir))
    with tqdm(total=test_sample_num, desc="test dataloader") as pbar:
        while True:
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data = next(data_iter)
            samples = vimeo2raw_dataloader2samples(data)
            for sample_id, sample in enumerate(samples):
                save_data_to = os.path.join(save_dir, "S{:06d}_{}".format(sample_id + iters * batch_size,
                                                                          sample["sample_name"]))
                DatasetVimeo2Raw.save_data(sample, save_data_to)
                pbar.update(1)
            iters += 1
            if iters == max_iters:
                break


def test_date_iter():
    from camera import O2S

    # setting
    set_seed()
    batch_size = 16
    # dataset
    data_file = "F:/database/vimeo_png/datasets/vimeo_val_sn-200.yaml"
    record_bad_data = data_file.replace(".yaml", "_bad_data.txt")
    dataset = DatasetVimeo2Raw(data_file,
                               O2S(), frame_num=7, record_bad_data=record_bad_data)
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    # data iter
    data_iter = iter(dataloader)

    # go through database
    samples = 0
    epoch = 0
    while True:
        try:
            data = next(data_iter)
            samples += data['sensor_raw'].shape[0]
            print("epoch: {}\ttotal sample: {}\t{}".format(epoch, samples, data['sensor_raw'].shape))
        except StopIteration:
            data_iter = iter(dataloader)
            data = next(data_iter)
            samples += data['sensor_raw'].shape[0]
            epoch += 1
            print("epoch: {}\ttotal sample: {}\t{}".format(epoch, samples, data['sensor_raw'].shape))
            if epoch == 2:
                break
    dataset.finish()


def check_database():
    from camera import O2S
    from tqdm import tqdm
    from aifactory.libs.log import ExperimentLogger

    data_file = "F:/database/vimeo_png/datasets/vimeo_val_sn-200.yaml"
    record_bad_data = data_file.replace(".yaml", "_bad_data.txt")
    log = ExperimentLogger(project_name="vimeo_dataset",
                           experiment_name=os.path.basename(data_file).replace(".yaml", ""),
                           config={},
                           log_dir=os.path.dirname(data_file),
                           use_trackio=True,
                           space_id=None)

    dataset = DatasetVimeo2Raw(data_file,
                               O2S(), frame_num=7, edge_mask_enable=True, record_bad_data=record_bad_data, log=log)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)
    for data in tqdm(dataloader):
        pass
    if hasattr(dataloader.dataset, "finish"):
        dataloader.dataset.finish()


if __name__ == "__main__":
    # test_dataset_vimeo2raw()
    # test_date_iter()
    check_database()

import os.path
from shutil import copyfile
import cv2
import numpy as np
import torch
from aifactory.utilts.load_file import load_file
from aifactory.libs.common.bgr2raw import bgr2raw
from aifactory.libs.common.raw2rgb import raw2bgr
from aifactory.utilts.save_files import save_as_image
from aifactory.utilts.seed import set_seed
from .raw_augment import add_noise_to_raw
from .camera import IspParameters, CAMERAS


class DatasetVimeo2Raw(torch.utils.data.Dataset):
    _dataset = None
    _cam = None
    _process_frame_num = None

    def __init__(self, paths, cam, frame_num=7):
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
            self._cam =CAMERAS[cam]()

        self._samples = list(self._dataset.keys())
        self._process_frame_num = frame_num
        self._item = None

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        self._item = item
        sample = self._dataset[self._samples[item]]
        # sample['files'] = np.array(sample['files'])
        frame_ids = self.get_sequence(sample)
        self._cam.get_tuning_parameters()

        frames = {"sensor_raw": [],
                  "gt": [],
                  "frame_id": []}
        for frame_id in frame_ids:
            bgr = cv2.imread(sample['files'][frame_id])
            raw = bgr2raw(bgr, self._cam)
            sensor_raw = self.add_noise(raw)
            frames["sensor_raw"].append(
                sensor_raw.astype(np.uint16) if self._cam.bits <= 16 else sensor_raw.astype(np.uint32))
            frames["gt"].append(raw.astype(np.uint16) if self._cam.bits <= 16 else raw.astype(np.uint32))
            frames["frame_id"].append(frame_id)
        frames["sensor_raw"] = np.array(frames["sensor_raw"])
        frames["gt"] = np.array(frames["gt"])
        frames["frame_id"] = np.array(frames["frame_id"])
        sample.update({"sample_name": self._samples[item],
                       "cam": self._cam.get_parameter_dict()
                       })
        sample.update(frames)
        return sample

    def get_sequence(self, sample):
        frame_num = sample['frame_num']
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

    def add_noise(self, raw):
        sensor_raw = add_noise_to_raw(raw.astype(np.float32) - self._cam.black_level,
                                      self._cam.bits, self._cam.read, self._cam.shot)
        return np.clip(sensor_raw["sensor_raw"] + self._cam.black_level, 0, self._cam.maximum).astype(np.uint16)

    '''
    def bit_align(self, raw):
        if self._bit_align != self._cam.bits:
            src_range = (1 << self._cam.bits) - 1
            dst_range = (1 << self._bit_align) - 1
            raw_align = np.round(raw / src_range * dst_range)
            read_align =  self._cam.read * (1 << (2 * (self._bit_align - self._cam.bits)))
            scale = (1 << (self._bit_align - self._cam.bits))
            shot_align = self._cam.shot * scale
            blacklevel_align = self._cam.blacklevel * scale
            return raw_align, shot_align, read_align, blacklevel_align
        else:
            return raw, self._cam.shot, self._cam.read, self._cam.blacklevel
    
    '''

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
                               O2S(), process_frame_num=5)
    save_dir = "H:/_result/ai_factory/raw_factory/DatasetVimeo2Raw"
    tqdm.write("\n{} test dataset {}\n save to:\t{}".format("*"*50, "*"*50, save_dir))
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
    dataset = DatasetVimeo2Raw("F:/database/vimeo_png/datasets/vimeo_val_sn-200.yaml",
                               O2S(), process_frame_num=5)
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    # data iter
    data_iter = iter(dataloader)
    # go throught
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


if __name__ == "__main__":
    # test_dataset_vimeo2raw()
    test_date_iter()

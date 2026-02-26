import random

import torch
import numpy as np
from aifactory.libs.data.camera.camera import IspParameters, CAMERAS
from aifactory.libs.data.control_scenes.simulated_images import SIMULATED_SENSES


class DatasetControlSense(torch.utils.data.IterableDataset):

    # frame information: resolution, channel, num
    _frame_height = None
    _frame_width = None
    _frame_channels = None
    _frame_num = None

    # parameters to convert a bgr frame to raw
    _cvt_to_raw = None
    _cam = None
    _iso = None
    _dst_bits = None

    # edge mask
    _edge_mask_enable = None
    _edge_mask_val = None

    # log
    _log = None

    def __init__(self,frame_height=256, frame_width=256, frame_channels=3, frame_num=1, cvt_to_raw=False,
                 cam=None, cvt_bits=None, iso=None, edge_mask_val=None, log=None):
        self._log = log

        self._frame_height = frame_height
        self._frame_width = frame_width
        self._frame_channels = frame_channels
        self._frame_num = frame_num

        self._cvt_to_raw = cvt_to_raw
        if cvt_to_raw:
            assert cam is not None, "if convert frame to raw, cam cannot be None"
            if isinstance(cam, str):
                assert cam in CAMERAS, "{} is not a recognized cammer".format(cam)
            self._cam = cam if isinstance(cam, IspParameters) else CAMERAS[cam]()
            self._dst_bits = cvt_bits
            self._iso = iso

        self._edge_mask_enable = edge_mask_val is not None
        self._edge_mask_val = edge_mask_val

    def __iter__(self):
        # Set independent and safe random seeds for each worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % 2 ** 32)
            torch.manual_seed(worker_info.seed)

        while True:
            try:
                # Generate simulated samples (replace with your actual simulation logic)
                sample = eval("self.{}()".format(random.choice(SIMULATED_SENSES)))
                yield sample
            except Exception as e:
                # Catch exception and print details (output to worker's stderr)
                print(f"Worker {worker_info.id if worker_info else 0} error: {e}")
                raise  # Re-raise to inform the main process of the crash cause

    def info(self, info):
        self._log.info(info) if (self._log is not None and hasattr(self._log, "info")) else print(info)

    def solid_color(self):
        self.info("generate solid color")
        image = torch.randn(self._frame_channels, self._frame_height, self._frame_width)
        return image

    def gray(self):
        self.info("generate gray")
        image = torch.randn(self._frame_channels, self._frame_height, self._frame_width)
        return image

    def stripe(self):
        self.info("generate stripe")
        image = torch.randn(self._frame_channels, self._frame_height, self._frame_width)
        return image

    def sin(self):
        self.info("generate sin")
        image = torch.randn(self._frame_channels, self._frame_height, self._frame_width)
        return image

    def edge(self):
        self.info("generate edge")
        image = torch.randn(self._frame_channels, self._frame_height, self._frame_width)
        return image


def test_dataset():
    dataset = DatasetControlSense()
    it = iter(dataset)
    for _ in range(100):
        sample = next(it)


if __name__ == "__main__":
    test_dataset()

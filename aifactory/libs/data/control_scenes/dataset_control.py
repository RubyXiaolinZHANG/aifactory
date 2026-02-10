import torch
from aifactory.libs.data.camera.camera import IspParameters, CAMERAS
from .simulated_images import SIMULATED_SENSES


class DatasetControlSense(torch.utils.data.Dataset):

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

    def __init__(self, cam, cvt_bits=None, crop=None, frame_num=1, iso=None, edge_mask_val=None,
                 record_bad_data=None, fix_sequence=False,
                 log=None):

        if isinstance(cam, IspParameters):
            self._cam = cam
        else:
            self._cam = CAMERAS[cam]()

        self._process_frame_num = frame_num
        self._iso = iso

        self._edge_mask_val = edge_mask_val
        self._edge_mask_enable = edge_mask_val is not None

    def __len__(self):
        return 1 << 16

    def __getitem__(self, item):
        pass

    
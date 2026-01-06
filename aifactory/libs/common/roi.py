import os.path
import cv2
import numpy as np
from .image import RawInfo
from .signal_processor import get_frequency_info, gradient2d
from .image_processor import enhanced_edge_detection, bgr2hsv
from .save_roi import add_pq_roi_method


class ROI:
    _x, _y, _w, _h = None, None, None, None

    def __init__(self, x, y, w, h):
        self._x = int(x)
        self._y = int(y)
        self._w = int(w)
        self._h = int(h)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def x1(self):
        return self._x

    @property
    def y1(self):
        return self._y

    @property
    def x2(self):
        return self._x + self._w

    @property
    def y2(self):
        return self._y + self._h

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def area(self):
        return self._w * self._h


@add_pq_roi_method
class ImageROI(ROI):
    _roi_image = None
    _src_image = None

    def __init__(self, image, x, y, w, h):
        self._src_image = image
        super().__init__(x, y, w, h)
        self._roi_image = self._src_image.crop(self)

    @property
    def src_image(self):
        return self._src_image

    @property
    def roi_image(self):
        return self._roi_image

    @property
    def c(self):
        return self._roi_image.c


class PQROI(ImageROI):
    _type = None
    _pq = None

    def __init__(self, image, roi):
        super().__init__(image, roi['coordinates']['x'], roi['coordinates']['y'], roi['coordinates']['w'],
                         roi['coordinates']['h'])
        self._type = roi['type']

    @property
    def type(self):
        return self._type

    @property
    def pq(self):
        return self._pq

    def __call__(self, *args, **kwargs):
        self.info()
        hist_inf = self.roi_image.hist(by_channel=False)
        getattr(self, "process_{}".format(self.type.replace(" ", "_")))()
        return self.pq

    def process_solid_color(self):
        mean_vals = self.roi_image.mean(by_channel=True)
        mse_val = ((self.roi_image.bayer - mean_vals) ** 2).mean()
        std_val = np.sqrt(mse_val)
        self._pq = {"type": self.type,
                    "signal": mean_vals,
                    "noise": std_val,
                    "snr": 10 * np.log10((mean_vals ** 2).mean() / mse_val),
                    "psnr": 20 * np.log10(self.roi_image.max_bound / std_val)}
        return self._pq

    def process_texture(self):
        self._pq = {"type": self.type,
                    "roughness": self.roi_image._histogram['roughness'],
                    "balance": self.roi_image._histogram['moment3']}
        return self._pq

    def process_grid(self, max_grid_size=4):

        def get_grid_energy(_fft):
            N = len(_fft['magnitude'])
            nequest = N // 2
            bandwidth = int(np.round(N * (0.5 - 1.0 / max_grid_size)))
            _fft['grid_frequcy_band'] = [nequest - bandwidth, nequest + bandwidth]
            h_freq = _fft['magnitude'][_fft['grid_frequcy_band'][0]: _fft['grid_frequcy_band'][1]]
            _fft['grid_energy'] = (h_freq ** 2).sum() / (_fft['magnitude'] ** 2).sum()

        # process sRGB, gray, y image; not available to Bayer
        if isinstance(self.roi_image, RawInfo):
            image = self.roi_image.demosaic()
            im_c = 3
        else:
            image = self.roi_image.image
            im_c = self.roi_image.c
        #
        if im_c == 1:
            h_project = image.mean(axis=0)
            v_project = image.mean(axis=1)
            h_freq = get_frequency_info(h_project)
            v_freq = get_frequency_info(v_project)
            h_grid_engery = h_freq['grid_energy']
            v_grid_engery = v_freq['grid_energy']

        else:
            h_project, v_project, h_freq, v_freq = [], [], [], []
            h_grid_engery, v_grid_engery = 0, 0
            for i in range(im_c):
                h_project.append(image[:, :, i].mean(axis=0))
                v_project.append(image[:, :, i].mean(axis=1))
                h_freq.append(get_frequency_info(h_project[-1]))
                get_grid_energy(h_freq[-1])
                h_grid_engery = h_freq[-1]['grid_energy'] if h_freq[-1][
                                                                 'grid_energy'] > h_grid_engery else h_grid_engery
                v_freq.append(get_frequency_info(v_project[-1]))
                get_grid_energy(v_freq[-1])
                v_grid_engery = v_freq[-1]['grid_energy'] if v_freq[-1][
                                                                 'grid_energy'] > v_grid_engery else v_grid_engery

        self._pq = {"h_prj": h_project,
                    "v_prj": v_project,
                    "h_frq": h_freq,
                    "v_frq": v_freq,
                    "h_grid_energy": h_grid_engery,
                    "v_grid_energy": v_grid_engery}
        return self._pq

    def process_chroma(self, h_threshold=15):
        if isinstance(self.roi_image, RawInfo):
            bgr = self.roi_image.demosaic()
        else:
            bgr = self.roi_image.image
        hsv = bgr2hsv(bgr)
        h, s, v = cv2.split(hsv)
        gradient_h = gradient2d(h)
        mask = np.abs(gradient_h['grad_x']) > h_threshold
        mask_h = mask[:, :-1] & mask[:, 1:]  # & mask[:,2 :-1 ]& mask[:,3 :]
        mask = np.abs(gradient_h['grad_y']) > h_threshold
        mask_v = mask[:-1, :] & mask[1:, :]  # & mask[2 :-1 , :]& mask[3 :, :]
        mask = np.zeros(h.shape, dtype=np.uint8)
        mask[:, :-1] = mask_h
        mask[:-1, :] = mask[:-1, :] | mask_v
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edge = enhanced_edge_detection(gray)
        width_edge = cv2.dilate(edge, np.ones((5, 5), dtype=np.uint8), iterations=1)
        chroma_mask = width_edge & (mask * 255)
        chroma_img = np.zeros(bgr.shape, dtype=np.uint8)
        chroma_img[:, :, 0] = chroma_mask & bgr[:, :, 0]
        chroma_img[:, :, 1] = chroma_mask & bgr[:, :, 1]
        chroma_img[:, :, 2] = chroma_mask & bgr[:, :, 2]
        fringing_ratio = (chroma_mask / 255).sum() / chroma_mask.size
        self._pq = {"src_image": bgr,
                    "fringing_ratio": fringing_ratio,
                    "fringing_image": chroma_img,
                    "fringing_mask": chroma_mask}
        # cv2.imwrite("chroma_image.png", chroma_img)
        return self._pq

    def info(self):
        print("Process {}\tx:{}\ty:{}\tw:{}\th:{}\t".format(self.type, self.x, self.y, self.w, self.h))

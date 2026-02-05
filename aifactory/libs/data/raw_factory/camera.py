import numpy as np

__all__ = ["CAMERAS"]

BLACK_LEVELS_BY_BITS = {10: 64,
                        11: 128,
                        12: 256,
                        13: 512,
                        14: 1024,
                        15: 2048,
                        16: 4096,
                        }


def noise_model(gain, _K_P0_, _K_P1_, _B_P0_, _B_P1_, _B_P2_):
    shot = _K_P0_ * gain + _K_P1_
    read = _B_P0_ * (gain ** 2) + _B_P1_ * gain + _B_P2_
    return shot, read


def ios2gain(ios):
    return ios / 50


def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class IspParameters:
    # hardware
    _sensor = None
    _bayer_pattern = None
    _bits = None
    _maximum = None
    _height = None
    _width = None

    # calibrate
    _shot = None
    _read = None

    # tuning
    _iso = None
    _black_level = None
    _analog_gain = None
    _d_gain = None
    _drc_gain = None
    _r_gain = None
    _g_gain = None
    _b_gain = None
    _ccm = None

    @property
    def sensor(self):
        return self._sensor

    @property
    def bayer_pattern(self):
        return self._bayer_pattern

    @property
    def bits(self):
        return self._bits

    @property
    def maximum(self):
        if self._bits is not None and self._maximum is None:
            self._maximum = (1 << self._bits) - 1
        return self._maximum

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    # calibrate
    @property
    def shot(self):
        return self._shot

    @property
    def read(self):
        return self._read

    # tuning
    @property
    def iso(self):
        return self._iso

    @property
    def black_level(self):
        return self._black_level

    @property
    def analog_gain(self):
        return self._analog_gain

    @property
    def d_gain(self):
        return self._d_gain

    @property
    def r_gain(self):
        return self._r_gain

    @property
    def g_gain(self):
        return self._g_gain

    @property
    def b_gain(self):
        return self._b_gain

    @property
    def ccm(self):
        return self._ccm

    def get_parameter_dict(self):
        property_type = (float, int, str, np.ndarray)
        properties = [attr for attr in dir(self) if (attr[:2] != "__" and
                                                     attr[0] == "_" and
                                                     isinstance(getattr(self, attr), property_type))]
        params = {}
        for p in properties:
            params[p[1:]] = getattr(self, p)
        return params

    def parse_meta(self, meta):
        pass


class O2S(IspParameters):
    # data control
    _min_ios = 40
    _max_ios = 3201
    _d_gain_center = 0.65
    _d_gain_scale = 0.17
    _d_gain_min = 1.0
    _d_gain_max = 8.0
    _default_ccm = np.array([[1.7079, -0.6175, -0.0904], [-0.4142, 1.6893, -0.2751], [-0.2399, -0.9428, 2.1827]],
                            dtype=np.float32)

    def __init__(self):
        self._sensor = 'O2S_main_OV50H'
        self._bayer_pattern = "RGGB"
        self._bits = 10
        self._maximum = 1023
        self._height = 2560
        self._width = 4096
        self._black_level = 64

        # noise of  10bit calibration
        self.RAWNF_CALIB = {
            '_K_P0_': 0.03024, '_K_P1_': 0.003186, '_K_P2_': 0.0, '_K_P3_': 0.0,
            '_B_P0_': 0.00101538, '_B_P1_': 0.00863597, '_B_P2_': 0.54664715
        }

        self.AWBs = {
            'D75': [379, 813],  # rg,bg
            'D65': [427, 738],  # rg,bg
            'D50': [475, 664],  # rg,bg
            'A': [723, 416],  # rg,bg
            'H': [928, 338],  # rg,bg
        }

        self.CCMs = {  # 3*6 的矩阵
            'D75': [1537, -476, -330, -426, 1563, -336, -202, -808, 1584, 320, -81, 53, 400, 7, -183, 383, 140, -72],
            'D65': [1537, -476, -330, -426, 1563, -336, -202, -808, 1584, 320, -81, 53, 400, 7, -183, 383, 140, -72],
            'D50': [1537, -476, -330, -426, 1563, -336, -202, -808, 1584, 320, -81, 53, 400, 7, -183, 383, 140, -72],
            'A': [1537, -476, -330, -426, 1563, -336, -202, -808, 1584, 320, -81, 53, 400, 7, -183, 383, 140, -72],
            'H': [1537, -476, -330, -426, 1563, -336, -202, -808, 1584, 320, -81, 53, 400, 7, -183, 383, 140, -72],
        }

    def get_noise(self, ios):
        return noise_model(ios, self.RAWNF_CALIB['_K_P0_'], self.RAWNF_CALIB['_K_P1_'],
                           self.RAWNF_CALIB['_B_P0_'], self.RAWNF_CALIB['_B_P1_'], self.RAWNF_CALIB['_B_P2_'])

    def get_awb(self):
        rand_index = np.random.randint(len(self.AWBs) - 1)
        awbs = (1 << self.bits) / np.array(list(self.AWBs.values()))
        weights = softmax(np.random.random(2))
        awb = awbs[rand_index] * weights[0] + awbs[rand_index + 1] * weights[1]
        r_gain = np.random.normal(loc=awb[0], scale=0.1)
        b_gain = np.random.normal(loc=awb[1], scale=0.1)
        g_gain = 1.0
        return max(r_gain, 0.8), g_gain, max(b_gain, 0.8)

    def get_ios(self):
        return np.random.randint(self._min_ios, self._max_ios)

    def get_digital_gain(self):
        return np.clip(1.0 / np.random.normal(loc=self._d_gain_center, scale=self._d_gain_scale),
                       self._d_gain_min, self._d_gain_max)

    def get_tuning_parameters(self, iso=None):
        self._iso = self.get_ios() if iso is None else iso
        self._analog_gain = ios2gain(self._iso)
        self._d_gain = self.get_digital_gain()
        self._r_gain, self._g_gain, self._b_gain = self.get_awb()
        self._ccm = np.eye(3)
        self._shot, self._read = self.get_noise(self._analog_gain)
        self._drc_gain = 1.0

    def clean_tuning_parameters(self):
        self._iso = None
        self._analog_gain = None
        self._d_gain = None
        self._r_gain = None
        self._g_gain = None
        self._b_gain = None
        self._ccm = None

    def parse_meta(self, meta):
        shot, read = self.get_noise(meta['exposure_metadata']['analog_gain'])

        raw_info = {"src": meta,
                    "cam": self.__class__.__name__,
                    "height": meta['frame_format']['resolution']['height'],
                    "width": meta['frame_format']['resolution']['width'],
                    'bayer_pattern': meta['frame_format']['cfa_pattern'].lower(),
                    "bits": meta["frame_format"]["bit_depth"],
                    "maximum": (1 << meta["frame_format"]["bit_depth"]) - 1,
                    "black_level": self._black_level << (meta["frame_format"]["bit_depth"] - self._bits),
                    "analog_gain": meta['exposure_metadata']['analog_gain'],
                    "d_gain": meta['exposure_metadata']['digital_gain'],
                    "drc_gain": meta['device_metadata']['DrcGain'],
                    "r_gain": meta['device_metadata']['gain_r'],
                    "g_gain": meta['device_metadata']['gain_g'],
                    "b_gain": meta['device_metadata']['gain_b'],
                    "ccm": np.array([[1.7079, -0.6175, -0.0904],
                                     [-0.4142, 1.6893, -0.2751],
                                     [-0.2399, -0.9428, 2.1827]],
                                    dtype=np.float32),
                    "read": read * (1 << (meta["frame_format"]["bit_depth"] - self._bits) * 2),
                    "shot": shot * (1 << (meta["frame_format"]["bit_depth"] - self._bits))}
        return raw_info


CAMERAS = {"O2S": O2S}

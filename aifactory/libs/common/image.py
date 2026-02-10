from copy import deepcopy

import cv2
import numpy as np
import os
from .mem_ops import space2depth, crop_image
from .raw2rgb import raw2bgr
from .statistics import histogram
from aifactory.libs.data.camera.camera import BLACK_LEVELS_BY_BITS


class ImageInfo:
    _image = None
    _image_pattern = None
    _image_height = None
    _image_width = None
    _image_channels = None
    _image_bits = None
    _histogram = None
    _histogram_by_channel = None

    def __init__(self, image, pattern, bits=8):
        self._image = image
        self._image_pattern = pattern
        if image.ndim == 2:
            self._image_height, self._image_width = image.shape
            self._image_channels = 1
        elif image.ndim == 3:
            self._image_height, self._image_width, self._image_channels = image.shape
        else:
            raise ValueError("dim of image should be 2 or 3 but got {}".format(image.ndim))
        self._image_bits = bits

    @property
    def image(self):
        return self._image

    @property
    def h(self):
        return self._image_height

    @property
    def w(self):
        return self._image_width

    @property
    def c(self):
        return self._image_channels

    @property
    def pattern(self):
        return self._image_pattern

    @property
    def dim(self):
        return self._image.ndim

    @property
    def bits(self):
        return self._image_bits

    @property
    def max_bound(self):
        return (1 << self._image_bits) - 1

    @property
    def histogram(self):
        return self._histogram

    @property
    def histogram_by_channel(self):
        return self._histogram_by_channel

    def crop(self, roi):
        return ImageInfo(crop_image(self.image, self._roi.y1, self._roi.y2, self._roi.x1, self._roi.x2),
                         self.pattern, self.bits)

    def hist(self, by_channel=False, image=None):
        axis = -1 if by_channel else None
        edge = np.arange(self.max_bound + 1)
        if image is None:
            if by_channel:
                self._histogram_by_channel = histogram(self.image, edge, axis=axis)
                return self._histogram_by_channel
            else:
                self._histogram = histogram(self.image, edge, axis=axis)
                return self._histogram
        else:
            return histogram(image, edge, axis=axis)

    def mean(self, by_channel=False):
        return self._image.mean(axis=(0, 1)) if by_channel else self._image.mean()

    def std(self, by_channel=False):
        return self._image.std(axis=(1, 2)) if by_channel else self._image.std()

    def hist_equalization(self, by_channel=False, image=None):
        if image is None:
            image = self._image
        if by_channel:
            if self._histogram_by_channel is None:
                self.hist(by_channel=by_channel)
            data = []
            for i in range(self.c):
                mapping = (self._histogram_by_channel[i]['cdf'] * self.max_bound).round()
                data.append(mapping[image[:, :, i]])
            # data = np.array(data)
        else:
            if self._histogram is None:
                self.hist(by_channel=by_channel)
            mapping = (self._histogram['cdf'] * self.max_bound).round()
            data = mapping[image]
        return data

    def save_image(self, save_name, image=None):
        if image is None:
            image = self.image
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        cv2.imwrite(save_name, (image.astype(np.float32) / self.max_bound * 255).round().astype(np.uint8))


class RawInfo(ImageInfo):
    _meta = None
    _bayer = None
    _bayer_height = None
    _bayer_width = None

    def __init__(self, image, meta):
        self._meta = meta
        if image.ndim == 1:
            image = image.reshape(meta['frame_format']['resolution']['height'],
                                  meta['frame_format']['resolution']['width'])
        super().__init__(image, pattern=meta['frame_format']['cfa_pattern'].lower(),
                         bits=meta['frame_format']["bit_depth"])

    @property
    def bayer_h(self):
        self._bayer_height = self.h // 2
        return None if self._bayer is None else self._bayer_height

    @property
    def bayer_w(self):
        self._bayer_width = self.w // 2
        return None if self._bayer is None else self._bayer_width

    @property
    def bayer_c(self):
        return None if self._bayer is None else 4

    @property
    def bayer(self):
        self._bayer = space2depth(self.image, 2, channels_last=True)
        return self._bayer

    @property
    def meta(self):
        return self._meta

    def mean(self, by_channel=False):
        return self.bayer.mean(axis=(0, 1)) if by_channel else self.bayer.mean()

    def std(self, by_channel=False):
        return self.bayer.std(axis=(1, 2)) if by_channel else self.bayer.std()

    def hist_equalization(self, by_channel=False, image=None):
        if image is None:
            image = self._bayer
        if by_channel:
            if self._histogram_by_channel is None:
                self.hist(by_channel=by_channel)
            data = []
            for i in range(self.c):
                mapping = (self._histogram_by_channel[i]['cdf'] * self.max_bound).round()
                data.append(mapping[image[:, :, i]])
        else:
            if self._histogram is None:
                self.hist(by_channel=by_channel)
            mapping = (self._histogram['cdf'] * self.max_bound).round()
            data = mapping[image]
        return data

    def crop(self, roi):
        data = crop_image(self._image, roi.y1 // 2 * 2, roi.y2 // 2 * 2, roi.x1 // 2 * 2, roi.x2 // 2 * 2)
        meta = deepcopy(self._meta)
        meta['frame_format']['resolution']["height"], meta['frame_format']['resolution']["width"] = data.shape[:2]
        return RawInfo(data, meta)

    def demosaic(self):
        ccm = np.array([[1.7079, -0.6175, -0.0904], [-0.4142, 1.6893, -0.2751], [-0.2399, -0.9428, 2.1827]],
                       dtype=np.float32)
        # ccm = np.array(self.meta['color_constancy']['ccm_srgb'])
        return raw2bgr(self.image, self.bits, self.pattern, BLACK_LEVELS_BY_BITS[self.bits],
                       self.meta['device_metadata']['gain_r'] * self.meta['exposure_metadata']['digital_gain'] * self.meta['device_metadata']['DrcGain'],
                       self.meta['device_metadata']['gain_g'] * self.meta['exposure_metadata']['digital_gain'] * self.meta['device_metadata']['DrcGain'],
                       self.meta['device_metadata']['gain_b'] * self.meta['exposure_metadata']['digital_gain'] * self.meta['device_metadata']['DrcGain'],
                       ccm=ccm,
                       )

    def save_bayer(self, save_name, bayer=None):
        if bayer is None:
            bayer = self._bayer
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        if bayer.ndim == 2:
            image = bayer
        elif bayer.shape[-1] == 4:
            image = np.vstack(
                [np.hstack([bayer[:, :, 0], bayer[:, :, 1]]), np.hstack([bayer[:, :, 2], bayer[:, :, 3]])])
        else:
            raise ValueError("Do not support saving bayer of shape {}".format(bayer.shape))
        cv2.imwrite(save_name, (image.astype(np.float32) / self.max_bound * 255).round().astype(np.uint8))

import os
import numpy as np
from aifactory.utils.save_files import save_as_image

TURN_ON_DEBUG = False


def inverse_gamma(x, gamma=2.2):
    return np.clip(x, 1e-8, None) ** gamma


def inverse_ccm2bgr(x, ccm):
    rgb2cam = np.linalg.inv(ccm)
    return np.matmul(x, rgb2cam)


def invert_gains(x, rgb_gain, r_gain, b_gain, inflection=0.9):
    gray = x.mean(axis=-1)
    mask = np.expand_dims((np.clip((gray - inflection), 0.0, None) / (1.0 - inflection)) ** 2.0, axis=-1)
    gains = np.ones(3) / rgb_gain
    # image patten H * W * BGR
    gains[0] /= b_gain
    gains[-1] /= r_gain
    safe_gains = mask + (1 - mask) * gains
    safe_gains[:, :, 0][safe_gains[:, :, 0] < gains[0]] = gains[0]
    safe_gains[:, :, 1][safe_gains[:, :, 1] < gains[1]] = gains[1]
    safe_gains[:, :, 2][safe_gains[:, :, 2] < gains[2]] = gains[2]
    return x * safe_gains


def mosaic(bgr, bayer_pattern):
    assert bgr.shape[-1] == 3
    if bayer_pattern.lower() == "bggr":
        return np.stack([bgr[0::2, 0::2, 0],
                         bgr[0::2, 1::2, 1],
                         bgr[1::2, 0::2, 1],
                         bgr[1::2, 1::2, 2]], axis=-1)

    elif bayer_pattern.lower() == "rggb":
        return np.stack([bgr[0::2, 0::2, 2],  # r
                         bgr[0::2, 1::2, 1],  # g
                         bgr[1::2, 0::2, 1],  # g
                         bgr[1::2, 1::2, 0]], axis=-1)  # b

    else:
        raise ValueError("Only support bayer format of bggr and rggb")


def channel2space(bayer):
    h, w, c = bayer.shape
    assert c == 4
    raw = np.zeros((h * 2, w * 2))
    raw[::2, ::2] = bayer[:, :, 0]
    raw[::2, 1::2] = bayer[:, :, 1]
    raw[1::2, ::2] = bayer[:, :, 2]
    raw[1::2, 1::2] = bayer[:, :, 3]
    return raw


def bgr2raw(bgr, cam, return_normalized_bgr=False):
    # normalize to [0, 1]
    norm_bgr = bgr.astype(np.float32) / 255
    if TURN_ON_DEBUG:
        save_as_image(norm_bgr, os.path.join("./_debug/test_images",
                                             "1_bgr_norm_[{:.2f}, {:.2f}].png".format(norm_bgr.min(),
                                                                                      norm_bgr.max())))
    # inverse gamma
    norm_bgr = inverse_gamma(norm_bgr, gamma=2.2)
    if TURN_ON_DEBUG:
        save_as_image(norm_bgr, os.path.join("./_debug/test_images",
                                             "2_bgr_inv_gamma_[{:.2f}, {:.2f}].png".format(norm_bgr.min(),
                                                                                           norm_bgr.max())))
    # inverse ccm
    norm_bgr = inverse_ccm2bgr(norm_bgr, cam.ccm)
    if TURN_ON_DEBUG:
        save_as_image(norm_bgr, os.path.join("./_debug/test_images",
                                             "3_bgr_inv_ccm_[{:.2f}, {:.2f}].png".format(norm_bgr.min(),
                                                                                         norm_bgr.max())))
    # inverse gains
    # print("inverse gains: d_gain:{}, r_gain:{}, b_gain:{}".format(cam.d_gain, cam.r_gain, cam.b_gain))
    norm_bgr = np.clip(invert_gains(norm_bgr, cam.rgb_gain, cam.r_gain, cam.b_gain), 0, 1) / cam.d_gain

    # scaling down and shift above black level
    black_level_rate = cam.black_level / cam.maximum
    norm_bgr = np.clip(norm_bgr * (1 - black_level_rate) + black_level_rate, 0.0, 1.0)
    if TURN_ON_DEBUG:
        save_as_image(norm_bgr, os.path.join("./_debug/test_images",
                                             "4_bgr_inv_gains_[{:.2f}, {:.2f}].png".format(norm_bgr.min(),
                                                                                           norm_bgr.max())))
    # mosaic
    norm_raw = mosaic(norm_bgr, cam.bayer_pattern)
    if TURN_ON_DEBUG:
        display_raw = np.vstack([np.hstack([norm_raw[:, :, 0], norm_raw[:, :, 1]]),
                                 np.hstack([norm_raw[:, :, 2], norm_raw[:, :, 3]])])
        save_as_image(display_raw, os.path.join("./_debug/test_images",
                                                "5_rggb_[{:.2f}, {:.2f}].png".format(norm_raw.min(),
                                                                                     norm_raw.max())))
        save_as_image(channel2space(norm_raw), os.path.join("./_debug/test_images",
                                                "5_bayer_[{:.2f}, {:.2f}].png".format(norm_raw.min(),
                                                                                     norm_raw.max())))
    if return_normalized_bgr:
        return channel2space(norm_raw)
    else:
        return  np.round(channel2space(norm_raw) * cam.maximum)
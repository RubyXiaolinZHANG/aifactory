import numpy as np
import cv2


def apply_gains2bgr(bgr, b_gain, g_gain, r_gain):
    bgr[:, :, 0] *= b_gain
    bgr[:, :, 1] *= g_gain
    bgr[:, :, 2] *= r_gain
    return bgr


def apply_ccm2bgr(bgr, ccm):
    return np.matmul(bgr, ccm.transpose())


def apply_gamma(x, gamma=2.2):
    return np.clip(x, 0, 1.0)**(1 / gamma)


def raw2bgr(raw, raw_bit, raw_pattern, black_level, r_gain, g_gain, b_gain, ccm=np.eye(3), gamma=2.2):
    # (1) convert data type to U16
    assert raw_bit <= 16
    max_bound = ((1 << raw_bit) - 1)
    raw = np.clip(raw, 0 , max_bound).astype(np.float32)

    # (2) black level compensation
    raw -= black_level
    raw = np.clip(raw, 0, max_bound).astype(np.uint16)

    # (3) raw to BGR
    if raw_pattern.lower() == "bggr" or raw_pattern.lower() == "bgbgrr":
        bgr = cv2.cvtColor(raw, cv2.COLOR_BAYER_BGGR2BGR)
    else:
        bgr = cv2.cvtColor(raw, cv2.COLOR_BAYER_RGGB2BGR)

    # (4) BGR normalization
    bgr = bgr.astype(np.float32)
    bgr /= max_bound

    # (5) apply gains
    bgr = np.clip(apply_gains2bgr(bgr, b_gain, g_gain, r_gain), 0, 1.0)

    # (6) apply ccm
    bgr = np.clip(apply_ccm2bgr(bgr, ccm), 0, 1.0)

    # (7) apply gamma
    bgr = apply_gamma(bgr, gamma)

    # (8) convert to standard BGR
    sBGR = np.clip(np.round(bgr * 255), 0, 255).astype(np.uint8)
    return sBGR


def raw2rgb(raw, raw_bit, raw_pattern, black_level, r_gain, g_gain, b_gain, ccm=np.eye(3), gamma=2.2):
    return cv2.cvtColor(raw2bgr(raw, raw_bit, raw_pattern, black_level, r_gain, g_gain, b_gain, ccm=ccm, gamma=gamma),
                        cv2.COLOR_BGR2RGB)

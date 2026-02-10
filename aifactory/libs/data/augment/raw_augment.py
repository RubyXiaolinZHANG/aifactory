import numpy as np


def add_noise_to_raw(raw, raw_bits, read, shot):
    # variance = np.clip(raw * shot + read, 0, 65535)
    variance = raw * shot + read  # np.clip(raw * shot + read, 0, 65535)
    # process light area
    # light_threshold = ((1 << raw_bits) - 1) * 0.93
    # variance[variance > light_threshold] = 0
    std = np.sqrt(variance)
    noise = np.random.normal(0, std)

    # add noise
    sensor_raw = raw + noise
    return {"sensor_raw": sensor_raw.round(),
            "raw": raw,
            "noise": noise }
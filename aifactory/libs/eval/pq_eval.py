from aifactory.libs.common.image import RawInfo
from aifactory.libs.common.roi import PQROI
from  aifactory.utils.load_file import load_file_raw


def raw_eval(frame_info):
    if frame_info['image']['bayer'].get("data") is None:
        frame_info['image']['bayer']["data"] = load_file_raw(frame_info['image']['bayer']["file"])
    raw = RawInfo(frame_info['image']['bayer']["data"], frame_info['image']['bayer']["meta"])
    # bgr = raw.demosaic()
    frame_info['image']['bayer']["image"] = raw
    for roi in frame_info["rois"]:
        if roi['coordinates'] is None:
            roi['coordinates'] = {"h": raw.h,
                                  "w": raw.w,
                                  "x": 0,
                                  "y": 0}
        img_roi = PQROI(raw, roi)
        img_roi()
        roi["pq"] = img_roi
    return frame_info



def rgb_eval(frame_info):
    pass

def raw_seq_eval(seq):
    pass


def rgb_seq_eval(seq):
    pass

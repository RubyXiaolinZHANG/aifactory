import numpy as np
from tqdm import tqdm
from aifactory.utils.get_files import get_target_files


src_dir = r"F:\database\AINR_V600\TestData\O2S\normal_video\O2S\raw10\N10103_250414_01_O2S_OVX9100_Portrait"

src_fils = get_target_files(src_dir, suffix=".raw")
im_h = 2560
im_w = 4096

accum_data = None
for file_id, file in tqdm(enumerate(src_fils)):
    data = np.fromfile(file, dtype=np.uint16)
    if accum_data is None:
        accum_data = data.astype(np.uint64)
    else:
        accum_data += data.astype(np.uint64)
    if file_id == 1:
        break

accum_data = accum_data // 2  # len(src_fils)
accum_data.astype(np.uint16).tofile("o2s/accum_2.raw")
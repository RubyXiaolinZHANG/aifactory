import yaml

noise_roi_1 = {"type": "solid color",
               "coordinates": {"x": 600, "y": 400, "w": 550, "h": 400},
               }

noise_roi_2 = {"type": "solid color",
               "coordinates": {"x": 2100, "y": 310, "w": 400, "h": 200},
               }

texture_roi_1 = {"type": "texture",
                 "coordinates": {"x": 2900, "y": 450, "w": 500, "h": 350},
                 }

texture_roi_2 = {"type": "texture",
                 "coordinates": {"x": 1500, "y": 2350, "w": 900, "h": 200},
                 }

grid_roi_1 = {"type": "grid",
              "coordinates": {"x": 600, "y": 400, "w": 550, "h": 400},
              }

grid_roi_2 = {"type": "grid",
              "coordinates": {"x": 2100, "y": 310, "w": 400, "h": 200},
              }

chroma_roi_1 = {"type": "chroma",
                "coordinates": None,
                }
image = {"bayer": {
    "file": 'D:/Program/Matlab/Matlab_PQ/test_images/N10103_250414_01_O2S_OVX9100_Portrait/wino_qat_iter_1500/frame_92_clean.raw',
    "patten": 'rggb',
    "meta": 'D:/Program/Matlab/Matlab_PQ/test_images/N10103_250414_01_O2S_OVX9100_Portrait/wino_qat_iter_1500/MF_0092.json'},
         "rgb": 'D:/Program/Matlab/Matlab_PQ/test_images/N10103_250414_01_O2S_OVX9100_Portrait/wino_qat_iter_1500/frame_92_clean_sRGB.png'}

data = {
    "image": image,
    "type": "video",
    "source": {
        "name": "N10103_250414_01_O2S_OVX9200_Portrait_wino_qat_iter_1500"
    },
    "rois": [noise_roi_1, noise_roi_2, texture_roi_1, texture_roi_2, grid_roi_1, grid_roi_2, chroma_roi_1],
}

yaml_file = data["image"]["bayer"]["file"].replace(".raw", ".yaml")
with open(yaml_file, 'w', encoding='utf-8') as f:
    yaml.dump(data, f, allow_unicode=True)

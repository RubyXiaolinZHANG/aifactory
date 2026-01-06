from copy import deepcopy
from aifactory.utilts.save_files import save_dict2yaml

# Dataset
dataset_vimeo = {"name": "vimeo",
                 "enable": True,
                 "path": "F:/database/vimeo_png/datasets/vimeo_val_sn-200.yaml",
                 "batch_size": 16}
dataset_control = {"name": "control",
                   "enable": False,
                   "batch_size": 4}
dataset_test = {"name": "o2s",
                "enable": True,
                "path": ['F:/database/AINR_V600/TestData/O2S/normal_video/O2S/']}
database = {"train": {dataset_vimeo["name"]: deepcopy(dataset_vimeo),
                      dataset_control["name"]: deepcopy(dataset_control)},
            "eval": {dataset_vimeo["name"]: deepcopy(dataset_vimeo),
                     dataset_control["name"]: deepcopy(dataset_control)},
            "test": {dataset_test["name"]: deepcopy(dataset_test),
                     dataset_control["name"]: {k: v for k, v in dataset_control.items() if k not in ["batch_size"]} }}

# train
optimizer = {"name": "adamW",
             "params": {"lr": 1e-3,
                        "betas": (0.9, 0.999),
                        "eps": 1e-8,
                        "weight_decay": 1e-2,
                        "amsgrad": False}}
train = {"ckpt": None,
         "path": "train",
         "optimizer": optimizer,
         "log": {"step": 10},
         "save": {"step": 100}}

# evaluate
eval = {"ckpt": None,
        "path": "eval",
        "metrics": None}

# inference
infer = {"ckpt": None,
         "path": "eval"}

save_root = "H:/_result/ai_factory/model_zoo/ainr/20260105"

config = {"task": "F3_AINR",
          "save_root": save_root,
          "database": database,
          "train": train,
          "eval": eval,
          "infer": infer
          }
yaml_file = "../configs/template.yaml"
save_dict2yaml(config, yaml_file, first_level_insert_space=True)
print("save config to: {}".format(yaml_file))

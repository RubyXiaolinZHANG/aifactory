import sys

sys.path.append("../../../")
sys.path.append("../")

import os
import argparse
from copy import deepcopy
import torch
from tqdm import tqdm
from aifactory.utils.load_file import load_file
from aifactory.utils.load_test_database import DATABASE_LOADERS
from aifactory.libs.data.raw_factory.camera import CAMERAS
from aifactory.utils.save_files import save_image
from core import AinrInference
from models import MODELS


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        # default="../configs/AINR_Ex.yaml",
                        # default="../configs/AINR_Dev.yaml",
                        default="../configs/AINR_Unet_Baseline.yaml",
                        help="path to database")
    return parser.parse_args()


def get_test_samples(params):
    databases = {}
    for database_name, database_setting in params.items():
        if not database_setting.get("enable", True):
            continue
        if "parameters" not in database_setting:
            samples = DATABASE_LOADERS[database_setting["database_loader"]]()
        elif isinstance(database_setting["parameters"], dict):
            samples = DATABASE_LOADERS[database_setting["database_loader"]](**database_setting["parameters"])
        elif isinstance(database_setting["parameters"], (list, tuple)):
            samples = DATABASE_LOADERS[database_setting["database_loader"]](*database_setting["parameters"])
        else:
            raise ValueError(
                "do not support database parameters of type: {}".format(type(database_setting["parameters"])))
        if len(samples) > 0:
            databases[database_name] = {"samples": samples,
                                        "cam": database_setting["cam"],
                                        "frame_num": database_setting["frame_num"]}
    if len(databases) > 0:
        return databases
    else:
        return None


def main(arguments):
    # get configurations
    config = load_file(arguments.config)

    # init device
    device = torch.device(config.get("device", "cpu"))

    # init model
    model = MODELS[config["model"]["name"]]()

    # init train data loaders
    samples = get_test_samples(config["database"]["test"])
    assert samples is not None

    # prepare log config
    config["log"]["log_dir"] = config["log"]["log_dir"].replace("\\", "/").replace("/train/", "/infer/")
    config["log"]["config"] = deepcopy(config)
    config["log"]["config"]["log"].pop("config")

    # init evaluator
    infer = AinrInference(model,
                          ckpt=config['infer']["ckpt"],
                          device=device,
                          log=config.get("log", None),
                          vis=config["eval"].get("vis", False),
                          dtype=eval(config['dtype']),
                          save_path=config["infer"]["path"]
                          )
    for database_name, database in samples.items():
        cam = CAMERAS[database['cam']]()
        for sample_id, (sample_name, sample) in enumerate(database["samples"].items()):
            sample_dir = os.path.join(config["infer"]["path"],
                                      database_name,
                                      sample_name).replace("\\",
                                                           "/")
            tqdm.write("[{}/{}] {} save to {}".format(sample_id,
                                                      len(database["samples"]),
                                                      sample_name,
                                                      sample_dir))
            with tqdm(total=min(database["frame_num"], len(sample)), desc="test dataset") as pbar:
                for frame_id, frame_info in sample.items():
                    if frame_id == database["frame_num"]:
                        break
                    else:
                        raw = load_file(frame_info['raw'])
                        meta = load_file(frame_info['meta'])
                        meta = cam.parse_meta(meta)
                        result = infer(raw, meta, work_mode="deploy")
                        # save result
                        for key, image in result['srgb'].items():
                            if isinstance(image, list):
                                image = image[0]
                            save_name = os.path.splitext(os.path.basename(frame_info['raw']))[0]
                            save_image(os.path.join(sample_dir, key, "{}.png".format(save_name)), image)
                    pbar.update(1)
            infer.reset()

    print("SUCCEEDED!")


if __name__ == "__main__":
    main(parse_arg())

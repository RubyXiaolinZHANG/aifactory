import argparse
import sys

sys.path.append("../../../")
sys.path.append("../")

from copy import deepcopy
import torch
from aifactory.utils.load_file import load_file
from aifactory.libs.data import init_dataloaders
from core import AinrEvaluator
from models import MODELS


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        default="../configs/AINR_Ex.yaml",
                        help="path to database")
    return parser.parse_args()


def main(arguments):
    # get configurations
    config = load_file(arguments.config)

    # init device
    device = torch.device(config.get("device", "cpu"))

    # init model
    model = MODELS[config["model"]["name"]]()

    # init train data loaders
    eval_data = init_dataloaders(config["database"]["eval"],
                                 num_workers=config["workers"])
    # prepare log config
    config["log"]["log_dir"] = config["log"]["log_dir"].replace("\\", "/").replace("/train/", "/eval/")
    config["log"]["config"] = deepcopy(config)
    config["log"]["config"]["log"].pop("config")

    # init evaluator
    evaluator = AinrEvaluator(model, eval_data,
                              metrics_params=config["eval"]["metrics"],
                              dominate_metrics=config["eval"]["dominate_metrics"],
                              ckpt=config['eval']["ckpt"],
                              device=device,
                              log=config.get("log", None),
                              log_interval=config["eval"]["intervals"]["log"],
                              vis=config["eval"].get("vis", False),
                              dtype=eval(config['dtype']),
                              save_path=config["eval"]["path"]
                              )

    evaluator()
    print("SUCCEEDED!")


if __name__ == "__main__":
    main(parse_arg())

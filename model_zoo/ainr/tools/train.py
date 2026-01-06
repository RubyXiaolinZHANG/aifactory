
import argparse
import sys
from copy import deepcopy
from winerror import SUCCEEDED

sys.path.append("../")
import torch
from aifactory.utilts.load_file import load_file
from aifactory.libs.data import init_dataloaders
from core import AinrTrainer
from models import MODELS


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        default="../configs/example.yaml",
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
    train_Data = init_dataloaders(config["database"]["train"],
                                 num_workers=config["workers"])
    # prepare log config
    config["log"]["config"] = deepcopy(config)
    config["log"]["config"]["log"].pop("config")

    # init trainer
    trainer = AinrTrainer(model, train_Data,
                          optimizer_params=config["train"]["optimizer"],
                          iterations=config["train"]["iters"],
                          device=device,
                          log=config.get("log", None),
                          log_interval=config["train"]["intervals"]["log"],
                          save_interval=config["train"]["intervals"]["save"]
                          )
    trainer()
    print("SUCCEEDED!")


if __name__ == "__main__":
    main(parse_arg())

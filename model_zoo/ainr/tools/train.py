import argparse
import os
import sys
import shutil
from copy import deepcopy

sys.path.append("../../../")
sys.path.append("../")
import torch
from aifactory.utils.load_file import load_file
from aifactory.utils.seed import set_seed
from aifactory.libs.data import init_dataloaders
from core import AinrTrainer, AinrEvaluator
from models import MODELS
torch.autograd.set_detect_anomaly(True)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        # default="../configs/AINR_Unet_Baseline.yaml",
                        default="../configs/AINR_Dev.yaml",
                        help="path to database")
    return parser.parse_args()


def main(arguments):
    # get configurations
    config = load_file(arguments.config)
    os.makedirs(config['save_root'], exist_ok=True)
    shutil.copy(arguments.config, config['save_root'])

    # init device
    device = torch.device(config.get("device", "cpu"))

    # init model
    set_seed(0)
    model = MODELS[config["model"]["name"]]()

    # init train data loaders
    train_data = init_dataloaders(config["database"]["train"],
                                  num_workers=config["workers"])
    # init evaluate data loader
    eval_data = init_dataloaders(config["database"]["eval"],
                                  num_workers=config["workers"])

    # prepare log config
    config["log"]["config"] = deepcopy(config)
    config["log"]["config"]["log"].pop("config")

    # init evaluator
    evaluator = AinrEvaluator(model, eval_data,
                              metrics_params=config["eval"]["metrics"],
                              dominate_metrics=config["eval"]["dominate_metrics"],
                              device=device,
                              vis=config["eval"].get("vis", False),
                              dtype=eval(config['dtype']),
                              save_path=config["eval"]["path"]
                              )

    # init trainer
    trainer = AinrTrainer(model, train_data,
                          optimizer_params=config["train"]["optimizer"],
                          loss_params=config["train"]["losses"],
                          iterations=config["train"]["iters"],
                          ckpt=config['train']["ckpt"],
                          evaluator=evaluator,
                          device=device,
                          log=config.get("log", None),
                          log_interval=config["train"]["intervals"]["log"],
                          save_interval=config["train"]["intervals"]["save"],
                          evaluate_before_train = config["train"].get("evaluate_before_train", False),
                          vis=config["train"].get("vis", False),
                          model_inputs=config["model"].get("model_inputs", None),
                          model_input_shapes=config["model"].get("input_shapes", None),
                          dtype=eval(config['dtype']),
                          save_path = config["train"]["path"]
                          )

    trainer()
    print("SUCCEEDED!")


if __name__ == "__main__":
    main(parse_arg())

import torch
from .raw_factory.dataset_video import DatasetVimeo2Raw

DATASSETS = {"DatasetVimeo2Raw": DatasetVimeo2Raw}


def init_dataloaders(config, num_workers=0):
    dataloaders = None
    for name, config in config.items():
        if config.get("enable", False):
            dataset = DATASSETS[config['dataset']](**config['parameters'])
            if dataloaders is None:
                dataloaders = {}
            dataloaders[name] = torch.utils.data.DataLoader(dataset,
                                                            batch_size=config['batch_size'],
                                                            num_workers=num_workers,
                                                            shuffle=config.get('shuffle', False))

        else:
            pass
    return dataloaders
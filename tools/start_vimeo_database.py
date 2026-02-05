import os
import argparse
from ruamel.yaml import YAML
from aifactory.utils.get_files import get_target_files
from aifactory.utils.random import split_dict_randomly


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database",
                        type=str,
                        default='F:/database/vimeo_png/sequences',
                        help="path to database")
    parser.add_argument("--save_to",
                        type=str,
                        default="F:/database/vimeo_png/datasets",
                        help="folder to save results")
    return parser.parse_args()


def main(arguments):

    database = arguments.database

    files = get_target_files(database, suffix=".png")
    vimeo = None
    for id, file in enumerate(files):
        file = file.replace("\\", "/")
        folders = file.split("/")
        sample_name = "vimeo-{}-{}".format(folders[-3], folders[-2])

        if vimeo is None:
            vimeo = {}

        if sample_name in vimeo:
            vimeo[sample_name]["files"].append(file)
            vimeo[sample_name]["frame_num"] = len(vimeo[sample_name]["files"])
        else:
            vimeo[sample_name] = {"id": len(vimeo),
                                   "database": "vimeo",
                                   "type": "video",
                                   "data_type": "bgr",
                                   "path": os.path.dirname(file),
                                   "frame_num": 1,
                                   "files":[file]}
    print("vimeo sample num: {}".format(len(vimeo)))

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    file_path = os.path.join(arguments.save_to, "vimeo-{}.yaml".format(len(vimeo))).replace("\\", "/")
    print("dump database to: {}".format(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(vimeo, f)

    exit()

    vimeo_train, vimeo_validate = split_dict_randomly(vimeo, ratio=0.99, seed=123)
    vimeo_train = {k: vimeo_train[k] for k in sorted(vimeo_train)}
    vimeo_validate = {k: vimeo_validate[k] for k in sorted(vimeo_validate)}

    # save vimeo trainset
    for name, data in zip(["train", "val"],[vimeo_train, vimeo_validate]):
        file_path = os.path.join(arguments.save_to, "vimeo_{}_sn-{}.yaml".format(name, len(data))).replace("\\", "/")
        print("dump {} database to: {}".format(name.upper(), file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)


if __name__ == "__main__":
    main(parse_arg())

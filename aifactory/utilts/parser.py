from .load_file import load_file


def sort_video_by_frame_id(dataset):
    for key, data in dataset.items():
        if data["type"] == 'video':
            data["frames"] = dict(sorted(data["frames"].items()))


def parse_low_level_task_evaluation_files(files):
    dataset = None
    for file in files:
        info = load_file(file)
        if info is None:
            continue
        if dataset is None:
            dataset = {}

        # load data
        if info['image'].get('bayer') is not None:
            # raw = np.fromfile(info['image']['bayer']['file'])
            meta = load_file(info['image']['bayer']['meta'])
            meta['file'] = info['image']['bayer']['meta']
            info['image']['bayer']['meta'] = meta
            # info['image']['bayer']['image'] = RawInfo(raw, meta)

        if info.get("type") == 'video':
            src = info.get("source")
            assert src is not None
            if src['name'] in dataset:
                dataset[src['name']]["frames"][src['frame_id']] = info
                dataset[src['name']]["frame_num"] += 1
            else:
                dataset[src['name']] = {"type":'video',
                                        "frames": {src['frame_id']: info},
                                        "frame_num": 1}
    sort_video_by_frame_id(dataset)
    return dataset
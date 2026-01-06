import os
import argparse
from aifactory.utilts.get_files import get_target_files
from aifactory.utilts.parser import parse_low_level_task_evaluation_files
from aifactory.libs.eval.pq_eval import raw_eval


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database",
                        type=str,
                        default="D:/Program/Matlab/Matlab_PQ/test_images",
                        help="path to database")
    parser.add_argument("--result_path",
                        type=str,
                        default="H:/_result/ai_factory/tool_test_pq",
                        help="folder to save results")
    return parser.parse_args()


def save_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for roi_id, roi in enumerate(results['rois']):
        roi['pq'].save_result(os.path.join(save_dir,
                                           "roi_{}_{}".format(roi_id, roi['type'].replace(" ", "_"))).replace("\\",
                                                                                                              "/"))


def main(arguments):
    # get configs
    database_path = arguments.database
    result_path = arguments.result_path

    # get test datas
    files = get_target_files(database_path, suffix=".yaml")
    dataset = parse_low_level_task_evaluation_files(files)

    for sample_name, sample in dataset.items():
        if 'N10103_250414_01_O2S_OVX9200_Portrait_wino_qat_iter_1500' != sample_name:
            continue
        if sample['type'] == 'video':
            for frame_id, frame in sample['frames'].items():
                result = raw_eval(frame)
                # save results
                save_results(result, os.path.join(result_path, sample_name, "frame_{:02d}".format(frame_id)))

        else:
            pass

    print("Succeed!")


if __name__ == "__main__":
    main(parse_arg())

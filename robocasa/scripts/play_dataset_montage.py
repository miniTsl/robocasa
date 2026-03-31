from robocasa.utils.dataset_registry import (
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)
from robocasa.utils.dataset_registry import get_ds_path
from robocasa.scripts.playback_dataset import playback_dataset

import os
import argparse
from termcolor import colored

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        help="path to store videos",
        default="/data/sunyi/robocasa/",
    )
    parser.add_argument(
        "--num_demos_per_task",
        type=int,
        help="number of demos to play per task",
        default=10,
    )
    args = parser.parse_args()

    output_dir = args.output
    # output_dir = os.path.expanduser(output_dir)
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    tasks = list(MULTI_STAGE_TASK_DATASETS) + list(SINGLE_STAGE_TASK_DATASETS)

    for task_i, task in enumerate(tasks):
        ds_path = get_ds_path(task, ds_type="human_im")

        parser = argparse.Namespace()
        parser.dataset = ds_path

        parser.render = False
        base_path = os.path.splitext(ds_path)[0]  # 去掉.hdf5扩展名
        base_dir = os.path.dirname(ds_path)  # 获取目录
        parser.video_path = base_dir
        parser.use_actions = False
        parser.use_abs_actions = False
        parser.render_image_names = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
        parser.use_obs = True
        parser.n = None
        parser.filter_key = None
        parser.video_skip = 1
        parser.first = False
        parser.verbose = False
        parser.extend_states = False
        parser.camera_height = 256
        parser.camera_width = 256

        print(
            colored(
                f"[{task_i+1} / {len(tasks)}] Playing sample demos for {task}", "yellow"
            )
        )
        playback_dataset(parser)
        print()
        print()

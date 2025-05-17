from functools import partial
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json

from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
# from octo.data.dataset import make_dataset_from_rlds
from jaxrl_m.data.oxe_dataset_utils import custom2_apply_frame_transforms, custom2_apply_trajectory_transforms, custom2_make_dataset_from_rlds
from octo.data.utils.data_utils import allocate_threads
from octo.utils.spec import ModuleSpec

from octo.data.oxe.oxe_standardization_transforms import custom_dataset_transform

default_config = {
    'name': None, 
    'data_dir': None, 
    'image_obs_keys': {"primary": "image", "wrist": "wrist_image"},
    # 'image_obs_keys': {"primary": "image", "wrist": None},
    'state_obs_keys': ["state"], 
    'language_key': 'language_instruction', 
    'absolute_action_mask': [False, False, False, False, False, False, True], 
    'action_normalization_mask': [True, True, True, True, True, True, False], 
    'action_proprio_normalization_type': 'normal', 
    'standardize_fn': custom_dataset_transform,
}

class EmbedDataset:
    """
    Args:
        data_mix: List of (dataset name, sampling weight) tuples, or a string specifying a pre-defined mix to
            load from `OXE_NAMED_MIXES`.
        data_dir: Base data directory that contains the datasets.
        train: whether this is a training or validation dataset.
        batch_size: batch size, if not provided output is not batched.
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        frame_transforms_threads: total number of parallel calls for frame transforms. If None, defaults to
            AUTOTUNE.
        act_pred_horizon: Number of steps to chunk the actions into.
    """

    def __init__(
        self,
        data_dir: str,
        data_mix: Union[str, Sequence[Tuple[str, float]]]=None,
        data_name: str=None,
        dataset_statistics_path: str=None,
        batch_size: Optional[int] = None,
        traj_transform_threads: Optional[int] = None,
        traj_read_threads: Optional[int] = None,
        frame_transforms_threads: Optional[int] = None,
        act_pred_horizon: int = 4,
        future_image_horizon: int = 8,
        window_size: int = 1,
        load_split='all'
    ):
        assert data_mix is not None or data_name is not None, "Either data_mix or data_name must be provided"

        if data_mix is not None:
            dataset_kwargs_list, _ = make_oxe_dataset_kwargs_and_weights(
                data_mix,
                data_dir,
                load_camera_views=("primary", "wrist"),
                load_proprio=True,
                load_language=True,
            )
        else:
            default_config["name"] = data_name
            default_config["data_dir"] = data_dir
            dataset_kwargs_list = [default_config]

        if dataset_statistics_path is not None:
            print('*'*20)
            print('Loading dataset statistics from', dataset_statistics_path)
            print('*'*20)
            with open(dataset_statistics_path, "r") as f:
                dataset_statistics = json.load(f)
            all_dataset_statistics = [dataset_statistics]
            self.dataset_sizes = [dataset_statistics["num_transitions"]]
        else:
            self.dataset_sizes = []
            all_dataset_statistics = []
            for dataset_kwargs in dataset_kwargs_list:
                _, dataset_statistics = custom2_make_dataset_from_rlds(**dataset_kwargs, load_split=load_split, num_parallel_reads=1, num_parallel_calls=1)
                self.dataset_sizes.append(dataset_statistics["num_transitions"])
                all_dataset_statistics.append(dataset_statistics)

        # allocate threads based on weights
        # threads_per_dataset = allocate_threads(traj_transform_threads, np.array([1.0] * len(dataset_kwargs_list)))
        # reads_per_dataset = allocate_threads(traj_read_threads, np.array([1.0] * len(dataset_kwargs_list)))

        # logging.info("Threads per dataset: %s", threads_per_dataset)
        # logging.info("Reads per dataset: %s", reads_per_dataset)

        # construct datasets
        datasets = []
        # for dataset_kwargs, dataset_statistics, threads, reads in zip(
        #     dataset_kwargs_list,
        #     all_dataset_statistics,
        #     threads_per_dataset,
        #     reads_per_dataset,
        # ):
        for dataset_kwargs, dataset_statistics in zip(
            dataset_kwargs_list,
            all_dataset_statistics,
        ):
            dataset, _ = custom2_make_dataset_from_rlds(
                **dataset_kwargs,
                shuffle=False,
                # num_parallel_calls=threads,
                # num_parallel_reads=reads,
                dataset_statistics=dataset_statistics,
                load_split=load_split,
                num_parallel_reads=1, num_parallel_calls=1
            )
            dataset = custom2_apply_trajectory_transforms(
                dataset,
                future_action_window_size=act_pred_horizon-1,
                future_image_window_size=future_image_horizon-1,
                window_size=window_size,
                # num_parallel_calls=threads,
                num_parallel_calls=1
            ).flatten(num_parallel_calls=1)
            datasets.append(dataset)

        dataset = datasets[0]
        for i in range(1, len(datasets)):
            dataset = dataset.concatenate(datasets[i])

        # apply frame transforms
        dataset = custom2_apply_frame_transforms(
            dataset,
            resize_size={
                "primary": (128, 128),  # workspace (3rd person) camera is at 256x256
                "wrist": (128, 128),  # wrist camera is at 128x128
            },
            num_parallel_calls=1,
        )

        # sequential batch (parallel batch seems to use much more memory)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)

        # this seems to reduce memory usage without affecting speed
        dataset = dataset.with_ram_budget(1)
        
        self.dlimp_dataset = dataset

    def iterator(self):
        return self.dlimp_dataset.iterator()
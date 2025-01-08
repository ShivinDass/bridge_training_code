from functools import partial
import inspect
import json
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from octo.data import traj_transforms, obs_transforms
from octo.data.utils.data_utils import (
    get_dataset_statistics,
    NormalizationType,
    tree_map,
)

def normalize_action_and_proprio(
    traj: dict, metadata: dict, normalization_type: NormalizationType
):
    """Normalizes the action and proprio fields of a trajectory using the given metadata."""
    # maps keys of `metadata` to corresponding keys in `traj`
    
    # keys_to_normalize = {
    #     "action": "actions",
    #     "proprio": "observation/proprio",
    # }
    keys_to_normalize = {
        "action": "action",
    }
    if "proprio" in traj["observation"]:
        keys_to_normalize["proprio"] = "observation/proprio"

    if normalization_type == NormalizationType.NORMAL:
        # normalize to mean 0, std 1
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get(
                "mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool)
            )
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x
                ),
            )
        return traj

    if normalization_type == NormalizationType.BOUNDS:
        # normalize to [-1, 1]
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get(
                "mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool)
            )
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(
                        2
                        * (x - metadata[key]["min"])
                        / (metadata[key]["max"] - metadata[key]["min"] + 1e-8)
                        - 1,
                        -1,
                        1,
                    ),
                    x,
                ),
            )
        return traj

    raise ValueError(f"Unknown normalization type {normalization_type}")

def custom2_make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Mapping[str, Optional[str]] = {},
    depth_obs_keys: Mapping[str, Optional[str]] = {},
    state_obs_keys: Sequence[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[Sequence[bool]] = None,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    load_split: str = 'all',
) -> Tuple[dl.DLataset, dict]:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the
    trajectory into a standard format, which includes the keys "observation" and "action". "observation"
    should be a dictionary containing some number of additional keys, which will be extracted into an even
    more standardized format according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in
    place of an old name to insert padding. For example, if after `standardize_fn`, your "observation" dict
    has RGB images called "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary":
    None, "wrist": "wrist"}`, then the resulting dataset will have an "observation" dict containing the keys
    "image_primary", "image_secondary", and "image_wrist", where "image_primary" corresponds to "workspace",
    "image_secondary" is a padding image, and "image_wrist" corresponds to "wrist".

    `state_obs_keys` is a list of 1-dimensional proprioceptive keys to concatenate into a single array, which
    will be placed in the "proprio" key of the "observation" dict. A single padding element (zero) will be
    inserted for each None entry.

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will
    contain the key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset,
            since one file usually contains many trajectories!).
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to
            extract from the "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in
            image_obs_keys.items()}`. If a value of `old` is None, inserts a padding image instead (empty
            string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        state_obs_keys (Sequence[str|None]): List of 1-dimensional proprioception keys to be extracted from
            the "observation" dict, concatenated, and mapped to "proprio". Inserts 1 element of padding (zero) for
            each None entry.
        language_key (str, optional): If provided, the "task" dict will contain the key
            "language_instruction", extracted from `traj[language_key]`.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. If `action_proprio_normalization_type` is "normal", this should contain "mean" and
            "std" keys. If `action_proprio_normalization_type` is "bounds", this should contain "min" and "max"
            keys. May also provide "num_transitions" and "num_trajectories" keys for downstream usage (e.g., for
            `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        absolute_action_mask (Sequence[bool], optional): By default, all action dimensions are assumed to be
            relative. This is important for when `future_action_window_size > 0`: actions that are taken
            from beyond the end of the trajectory (or beyond the goal timestep when goal relabeling is used)
            need to be made "neutral" to indicate that the task has been completed. For relative actions,
            "neutral" means zero, but for absolute actions, "neutral" means repeating the last valid action.
            This mask, if provided, indicates which action dimensions are absolute.
        action_normalization_mask (Sequence[bool], optional): If provided, indicates which action dimensions
            should be normalized. For example, you might not want to normalize the gripper action dimension if
            it's always exactly 0 or 1. By default, all action dimensions are normalized.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    def restructure(traj):
        # apply a standardization function, if provided
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. "
                "Did you write a `standardize_fn`?"
            )

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                    if key is None
                    else tf.cast(old_obs[key], tf.float32)
                    for key in state_obs_keys
                ],
                axis=1,
            )

        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # extracts `language_key` into the "task" dict
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {traj[language_key].dtype}, "
                    "but it must be tf.string."
                )
            task["language_instruction"] = traj.pop(language_key)

        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj

    # builder = tfds.builder(name, data_dir=data_dir)
    # shivin: use the followin when using downloaded datasets, otherwise tfds throws an error
    builder = tfds.builder(f"{name}:0.1.0", data_dir=data_dir)

    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads
        ).traj_map(restructure, num_parallel_calls)
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
            ),
            save_dir=builder.data_dir,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    # skip normalization for certain action dimensions
    if action_normalization_mask is not None:
        if (
            len(action_normalization_mask)
            != dataset_statistics["action"]["mean"].shape[-1]
        ):
            raise ValueError(
                f"Length of skip_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)
    
    assert load_split in ['all', 'train', 'val'], f"Invalid load_split: {load_split}"
    # construct the dataset
    if load_split == 'all':
        split = "all"
    elif "val" not in builder.info.splits:
        split = "train[:95%]" if load_split=='train' else "train[95%:]"
    else:
        split = "train" if load_split=='train' else "val"
    print(f"Using split: {split}")

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )

    dataset = dataset.traj_map(restructure, num_parallel_calls)
    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )

    return dataset, dataset_statistics

def custom2_apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    future_action_window_size: int,
    future_image_window_size: int,
    window_size: int = 1,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a trajectory level. Such transforms are usually some sort of
    "relabeling" (e.g. filtering, chunking, adding goals, dropping keys). Transforms that happen in this
    function should have the following properties:

    - They require access to an entire trajectory (i.e. they cannot be applied in a frame-wise manner).
    - They are generally not CPU-intensive, mostly involving moving and copying data.
    - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        future_action_window_size (int, optional): The number of future actions beyond window_size to include
            in the chunked actions.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    # adds future observations to the trajectory
    def add_future_obs(traj):
        traj_len = tf.shape(traj["action"])[0]
        future_image_indices = tf.minimum(tf.range(traj_len) + future_image_window_size + 1, traj_len - 1)
        traj["future_image"] = tf.gather(traj["observation"]["image_primary"], future_image_indices)
        return traj
    
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)
    
    dataset = dataset.traj_map(
        add_future_obs,
        num_parallel_calls,
    )

    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )

    def create_image_embedding_copies(traj):
        traj["image_primary_encoding"] = traj["observation"]["image_primary"]
        traj["image_wrist_encoding"] = traj["observation"]["image_wrist"]
        return traj
    
    dataset = dataset.traj_map(
        create_image_embedding_copies,
        num_parallel_calls,
    )
    
    return dataset

def custom2_apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    resize_size: Tuple[int, int],
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a frame level. These transforms are usually more
    CPU-intensive, (e.g. decoding or resizing images).

    Args:
        dataset (dl.DLataset): The dataset to transform.
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        num_parallel_calls (int): number of parallel calls for frame_map operations. Default to AUTOTUNE.
    """

    def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
        # task is not chunked -- apply fn directly
        frame["task"] = fn(frame["task"])
        # observation is chunked -- apply fn along first axis
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    # decode + resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(
                obs_transforms.decode_and_resize,
                resize_size=resize_size,
                depth_resize_size={},
            ),
        ),
        num_parallel_calls,
    )

    # decode + resize images
    def decode_and_resize_future(obs, resize_size):
        future_image = obs["future_image"]
        if future_image.dtype == tf.string:
            if tf.strings.length(future_image) == 0:
                future_image = tf.zeros((*resize_size, 3), dtype=tf.uint8)
            else:
                future_image = tf.io.decode_image(
                    future_image, expand_animations=False, dtype=tf.uint8
                )
        elif future_image.dtype != tf.uint8:
            raise ValueError(
                f"Unsupported image dtype: found future image with dtype {future_image.dtype}"
            )
        future_image = dl.transforms.resize_image(future_image, size=resize_size)
        obs["future_image"] = future_image
        return obs

    dataset = dataset.frame_map(
        partial(
            decode_and_resize_future,
            resize_size=resize_size['primary'],
        ),
        num_parallel_calls,
    )

    return dataset

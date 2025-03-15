from functools import partial
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from jaxrl_m.data.oxe_dataset_utils import custom2_make_dataset_from_rlds
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.data.utils.data_utils import allocate_threads, tree_map, NormalizationType
from octo.utils.spec import ModuleSpec


def normalize_action_and_proprio(
    traj: dict, metadata: dict, normalization_type: NormalizationType
):
    """Normalizes the action and proprio fields of a trajectory using the given metadata."""
    # maps keys of `metadata` to corresponding keys in `traj`

    keys_to_normalize = {
        "action": "actions",
    }
    if "proprio" in traj["observations"]:
        keys_to_normalize["proprio"] = "observations/proprio"

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

    raise ValueError(f"Unknown normalization type {normalization_type}")

def custom_make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    image_obs_keys: Mapping[str, Optional[str]] = {},
    filter_functions: Sequence[ModuleSpec] = (),
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    dataset_statistics: Optional[dict] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    **kwargs,
) -> dl.DLataset:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the
    trajectory into a standard format, which includes the keys "observation" and "action". "observation"
    should be a dictionary containing some number of additional keys, which will be extracted into an even
    more standardized format according to the "*_obs_keys" arguments.

    The `image_obs_keys` argument is a mapping from new names to old names, or None in
    place of an old name to insert padding. For example, if after `standardize_fn`, your "observation" dict
    has RGB images called "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary":
    None, "wrist": "wrist"}`, then the resulting dataset will have an "observation" dict containing the keys
    "image_primary", "image_secondary", and "image_wrist", where "image_primary" corresponds to "workspace",
    "image_secondary" is a padding image, and "image_wrist" corresponds to "wrist".

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to
            extract from the "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in
            image_obs_keys.items()}`. If a value of `old` is None, inserts a padding image instead (empty
            string).
        filter_functions (Sequence[ModuleSpec]): ModuleSpecs for filtering functions applied to the
            raw dataset.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
        - action                        # action vector
    """
    REQUIRED_KEYS = {"observation", "action"}

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
        if image_obs_keys["primary"] is None:
            new_obs["image"] = tf.repeat("", traj_len) # padding
        else:
            new_obs["image"] = old_obs[image_obs_keys["primary"]]

        traj = {
            "observations": new_obs,
            "actions": tf.cast(traj["action"], tf.float32),
        }

        return traj

    builder = tfds.builder(name, data_dir=data_dir)

    dataset_statistics = tree_map(np.array, dataset_statistics)

    # construct the dataset
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=False, num_parallel_reads=num_parallel_reads
    )
    for filter_fcn_spec in filter_functions:
        dataset = dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
    dataset = dataset.traj_map(restructure, num_parallel_calls)
    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )

    return dataset


def custom_apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    future_action_window_size: int = 0,
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
        traj_len = tf.shape(traj["actions"])[0]
        future_image_indices = tf.minimum(tf.range(traj_len) + future_action_window_size + 1, traj_len - 1)
        traj["future_image"] = tf.gather(traj["observations"]["image"], future_image_indices)
        return traj
    
    dataset = dataset.traj_map(
        add_future_obs,
        num_parallel_calls,
    )

    # chunks actions, giving it a new axis at index 1 of size `1 + future_action_window_size`
    def chunk_act(traj, future_action_window_size: int = 0):
        traj_len = tf.shape(traj["actions"])[0]
        action_chunk_indices = tf.broadcast_to(
            tf.range(1 + future_action_window_size), [traj_len, 1 + future_action_window_size],
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None], [traj_len, 1 + future_action_window_size],
        )
        action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
        traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
        return traj
    
    dataset = dataset.traj_map(
        partial(
            chunk_act,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )
    
    return dataset


def custom_apply_frame_transforms(
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

    # decode + resize images
    def decode_and_resize(obs, resize_size):
        image = obs["observations"]["image"]
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                image = tf.zeros((*resize_size, 3), dtype=tf.uint8)
            else:
                image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
        elif image.dtype != tf.uint8:
            raise ValueError(
                f"Unsupported image dtype: found image with dtype {image.dtype}"
            )
        image = dl.transforms.resize_image(image, size=resize_size)
        obs["observations"]["image"] = image

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
            decode_and_resize,
            resize_size=resize_size,
        ),
        num_parallel_calls,
    )

    return dataset


class OXEGMFlowDataset:
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
        data_mix: Union[str, Sequence[Tuple[str, float]]],
        data_dir: str,
        train: bool,
        batch_size: Optional[int] = None,
        traj_transform_threads: Optional[int] = None,
        traj_read_threads: Optional[int] = None,
        frame_transforms_threads: Optional[int] = None,
        act_pred_horizon: int = 1,
    ):
        dataset_kwargs_list, _ = make_oxe_dataset_kwargs_and_weights(
            data_mix,
            data_dir,
            load_proprio=False,
            load_language=False,
        )

        real_dataset_kwargs_list = []
        for dataset_kwargs in dataset_kwargs_list:
            if dataset_kwargs["image_obs_keys"]["primary"] is None:
                continue
            real_dataset_kwargs_list.append(dataset_kwargs)
        
        dataset_sizes = []
        all_dataset_statistics = []
        for dataset_kwargs in dataset_kwargs_list:
            _, dataset_statistics = custom2_make_dataset_from_rlds(**dataset_kwargs)
            dataset_sizes.append(dataset_statistics["num_transitions"])
            all_dataset_statistics.append(dataset_statistics)

        # allocate threads based on weights
        threads_per_dataset = allocate_threads(traj_transform_threads, np.array([1.0] * len(real_dataset_kwargs_list)))
        reads_per_dataset = allocate_threads(traj_read_threads, np.array([1.0] * len(real_dataset_kwargs_list)))

        logging.info("Threads per dataset: %s", threads_per_dataset)
        logging.info("Reads per dataset: %s", reads_per_dataset)

        # construct datasets
        datasets = []
        for dataset_kwargs, dataset_statistics, threads, reads in zip(
            real_dataset_kwargs_list,
            all_dataset_statistics,
            threads_per_dataset,
            reads_per_dataset,
        ):
            dataset = custom_make_dataset_from_rlds(
                **dataset_kwargs,
                train=train,
                num_parallel_calls=threads,
                num_parallel_reads=reads,
                dataset_statistics=dataset_statistics,
            )
            dataset = custom_apply_trajectory_transforms(
                dataset,
                future_action_window_size=act_pred_horizon-1,
                num_parallel_calls=threads,
            ).flatten(num_parallel_calls=threads)
            datasets.append(dataset)

        dataset = datasets[0]
        for i in range(1, len(datasets)):
            dataset = dataset.concatenate(datasets[i])

        # apply frame transforms
        dataset = custom_apply_frame_transforms(
            dataset,
            resize_size=(128, 128),
            num_parallel_calls=frame_transforms_threads,
        )

        # sequential batch (parallel batch seems to use much more memory)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)

        # this seems to reduce memory usage without affecting speed
        dataset = dataset.with_ram_budget(1)
        
        self.dlimp_dataset = dataset

    def iterator(self):
        return self.dlimp_dataset.iterator()
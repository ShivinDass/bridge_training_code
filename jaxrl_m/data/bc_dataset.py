"""
tf.data.Dataset based dataloader for the BridgeData format, meaning
TFRecords with one trajectory per example. See the BridgeDataset class
below for more details.

Written by Kevin Black (kvablack@berkeley.edu).
"""

import fnmatch
from typing import Iterable, List, Optional, Union
from functools import partial

import numpy as np
import tensorflow as tf
from absl import logging

from jaxrl_m.data.tf_augmentations import augment, random_resized_crop
from jaxrl_m.data.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS


def glob_to_path_list(
    glob_strs: Union[str, List[str]], prefix: str = "", exclude: Iterable[str] = ()
):
    """Converts a glob string or list of glob strings to a list of paths."""
    if isinstance(glob_strs, str):
        glob_strs = [glob_strs]
    path_list = []
    for glob_str in glob_strs:
        paths = tf.io.gfile.glob(f"{prefix}/{glob_str}")
        filtered_paths = []
        for path in paths:
            if not any(fnmatch.fnmatch(path, e) for e in exclude):
                filtered_paths.append(path)
            else:
                logging.info(f"Excluding {path}")
        assert len(filtered_paths) > 0, f"{glob_str} came up empty"
        path_list += filtered_paths
    return path_list


@tf.function(jit_compile=True)
def _binarize_gripper_actions(actions):
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near
    1.0) or fully closed (near 0.0). As it transitions between the two, it
    sometimes passes through a few intermediate values. We relabel those
    intermediate values based on the state that is reached _after_ those
    intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we
    give up on binarizing and relabel that chunk of intermediate values as
    the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, tf.float32),
            lambda: is_open_float[i],
        )

    new_actions = tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )
    return new_actions


class ImgBCDataset:
    """
    Fast parallel tf.data.Dataset-based dataloader for a dataset in the
    BridgeData format. This format consists of TFRecords where each example
    is one trajectory. See `PROTO_TYPE_SPEC` below for the expected format
    for each example in more detail. See `_process_trajectory` below for
    the output format.

    Includes goal relabeling, image augmentations, and sampling from multiple
    datasets with different weights. Goal relabeling uses a 0/-1 reward scheme:
    0 when the next_obs is labeled as the goal, -1 otherwise.

    Args:
        data_paths: List of paths to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        seed: Random seed.
        action_proprio_metadata: Dictionary containing metadata of the actions and proprio.
            If provided, actions and proprio will be normalized.
        normalization_type: The type of normalization to apply to the actions
            and proprio.
        relabel_actions: Whether to relabel the actions with reached states
            (based on proprioception). Also binarizes gripper actions.
        sample_weights: If data_paths is a list of list of paths, this is a
            list of weights with which to sample from each sub-list.
        batch_size: Batch size.
        shuffle_buffer_size: Size of the shuffle buffer. It is split between
            sub-datasets by `sample_weights`.
        train: Whether this dataset is intended for training
            (if set to `False`, will disable shuffling and augmentations).
        augment: Whether to apply image augmentations.
        augment_kwargs: Keyword arguments for image augmentations. See
            `jaxrl_m.data.tf_augmentations.augment` for more details.
        act_pred_horizon: Number of consecutive actions that will be predicted.
        obs_horizon: Number of consecutive observations that will be conditioned on.
        goal_relabeling_strategy: Goal relabeling strategy. See
            `jaxrl_m.data.tf_goal_relabeling` for more details.
        goal_relabeling_kwargs: Keyword arguments for goal relabeling. See
            `jaxrl_m.data.tf_goal_relabeling` for more details.
        augment_next_obs_goal_differently: Whether to use different random seeds
            for augmenting the obs, next_obs, and goal image.
        load_langauge: Whether to look for and load language from the data.
        skip_unlabeled: Whether to filter out trajectories not labeled with language.
    """

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        action_proprio_metadata: Optional[dict] = None,
        normalization_type: Optional[str] = "normal",
        relabel_actions: bool = True,
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 10000,
        train: bool = True,
        augment: bool = False,
        augment_kwargs: dict = {},
        act_pred_horizon: Optional[int] = None,
        prechunk_act: bool = False,
        obs_horizon: Optional[int] = None,
        goal_conditioned: bool = False,
        goal_relabeling_strategy: str = "uniform",
        goal_relabeling_kwargs: dict = {},
        augment_goal_differently: bool = False,
        load_language: bool = False,
        skip_unlabeled: bool = False,
        included_in_action_loss: Optional[List[bool]] = None,
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to BridgeDataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        if included_in_action_loss is None:
            included_in_action_loss = [True] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert len(data_paths) == len(included_in_action_loss)
        assert np.isclose(sum(sample_weights), 1.0)

        self.relabel_actions = relabel_actions
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        self.augment_kwargs = augment_kwargs
        self.act_pred_horizon = act_pred_horizon
        self.prechunk_act = prechunk_act
        self.obs_horizon = obs_horizon
        self.is_train = train
        self.goal_conditioned = goal_conditioned
        self.goal_relabeling_strategy = goal_relabeling_strategy
        self.goal_relabeling_kwargs = goal_relabeling_kwargs
        self.augment_goal_differently = augment_goal_differently
        self.load_language = load_language

        assert not (self.prechunk_act and self.relabel_actions), "Cannot prechunk and relabel actions"
        if self.relabel_actions:
            self.PROTO_TYPE_SPEC["observations/state"] = tf.float32
            self.PROTO_TYPE_SPEC["next_observations/state"] = tf.float32
        if self.load_language:
            self.PROTO_TYPE_SPEC["language"] = tf.string
        if self.act_pred_horizon is not None:
            self.goal_relabeling_kwargs["act_pred_horizon"] = self.act_pred_horizon

        # construct a dataset for each sub-list of paths
        datasets = []
        for i, sub_data_paths in enumerate(data_paths):
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed, included_in_action_loss[i]))

        if train:
            # shuffle and repeat each sub-dataset, allocating the shuffle buffer
            # by sample_weights
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
                    .repeat()
                )

        # for validation, we want to be able to iterate through the entire dataset;
        # for training, we want to make sure that no sub-dataset is ever exhausted
        # or the sampling ratios will be off. this should never happen because of the
        # repeat() above, but `stop_on_empty_dataset` is a safeguard
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=train
        )

        if skip_unlabeled:
            dataset = dataset.filter(
                lambda x: tf.math.reduce_any(x["goals"]["language"] != "")
            )

        if train and augment:
            # apply augmentations, using a sequence of integers as seeds.
            # this was the only way I found to avoid a memory leak in tf.random.Generator
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        # NOTE: this is just a hack for viper_x data
        if train:
            dataset = dataset.batch(
                batch_size,
                num_parallel_calls=tf.data.AUTOTUNE,
                drop_remainder=True,
                deterministic=not train,
            )
        else:
            dataset = dataset.batch(
                batch_size,
                num_parallel_calls=tf.data.AUTOTUNE,
                drop_remainder=False,
                deterministic=not train,
            )

        self.tf_dataset = dataset

    def _construct_tf_dataset(self, paths: List[str], seed: int, included_in_action_loss: bool) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """

        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._process_actions, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._chunk_act_obs, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)
        
        # yields trajectories
        dataset = dataset.map(
            partial(self._add_action_loss_mask, mask_value=included_in_action_loss),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # unbatch to yield individual transitions
        # NOTE: unbatch is slow
        # dataset = dataset.unbatch()
        dataset = dataset.interleave(tf.data.Dataset.from_tensor_slices, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "actions": tf.float32,
        "image_flows": tf.float16,
    }

    def _decode_example(self, example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }
        # restructure the dictionary into the downstream format
        return {
            "observations": {
                "image": parsed_tensors["observations/images0"],
                **({"proprio": parsed_tensors["observations/state"]} if self.relabel_actions else {}),
            },
            "next_observations": {
                **({"proprio": parsed_tensors["next_observations/state"]} if self.relabel_actions else {}),
            },
            **({"language": parsed_tensors["language"]} if self.load_language else {}),
            "actions": parsed_tensors["actions"],
            "image_flows": parsed_tensors["image_flows"],
        }

    def _process_actions(self, traj):
        if self.relabel_actions:
            # relabel the first 6 action dims (xyz position, xyz rotation)
            # using the reached proprio
            movement_actions = (
                traj["next_observations"]["proprio"][:, :6]
                - traj["observations"]["proprio"][:, :6]
            )
            # binarize the gripper action
            continuous_gripper_actions = traj["actions"][:, 6]
            binarized_gripper_actions = _binarize_gripper_actions(
                continuous_gripper_actions
            )

            traj["actions"] = tf.concat(
                [movement_actions, binarized_gripper_actions[:, None]], axis=1
            )

        # normalize actions and proprio
        # NOTE: Not sure whether the action_proprio_meta we're using matches the relabeled actions
        # NOTE: We binarize the gripper actions but then normalize them, which is weird
        # NOTE: We don't use proprio for this dataset
        if self.action_proprio_metadata is not None:
            if self.normalization_type == "normal":
                # normalize to mean 0, std 1
                traj["actions"] = (
                    traj["actions"] - self.action_proprio_metadata["action"]["mean"]
                ) / self.action_proprio_metadata["action"]["std"]
                # if "proprio" in traj["observations"]:
                #     traj["observations"]["proprio"] = (
                #         traj["observations"]["proprio"]
                #         - self.action_proprio_metadata["proprio"]["mean"]
                #     ) / self.action_proprio_metadata["proprio"]["std"]
            elif self.normalization_type == "bounds":
                # normalize to [0, 1]
                traj["actions"] = (
                    traj["actions"] - self.action_proprio_metadata["action"]["min"]
                ) / (
                    self.action_proprio_metadata["action"]["max"]
                    - self.action_proprio_metadata["action"]["min"]
                )
                # clip to [0, 1]
                traj["actions"] = tf.clip_by_value(traj["actions"], 0, 1)
                # if "proprio" in traj["observations"]:
                #     traj["observations"]["proprio"] = (
                #         traj[key]["proprio"]
                #         - self.action_proprio_metadata["proprio"]["min"]
                #     ) / (
                #         self.action_proprio_metadata["proprio"]["max"]
                #         - self.action_proprio_metadata["proprio"]["min"]
                #     )
                #     traj["observations"]["proprio"] = tf.clip_by_value(traj["observations"]["proprio"], 0, 1)
            else:
                raise ValueError

        return traj

    def _chunk_act_obs(self, traj):
        traj_len = len(traj["actions"])
        if self.act_pred_horizon is not None and not self.prechunk_act:
            chunk_indices = tf.broadcast_to(
                tf.range(self.act_pred_horizon), [traj_len, self.act_pred_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.act_pred_horizon]
            )
            # pads by repeating the last action
            chunk_indices = tf.minimum(chunk_indices, traj_len - 1)
            traj["action_chunks"] = tf.gather(traj["actions"], chunk_indices)
        if self.obs_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(-self.obs_horizon + 1, 1), [traj_len, self.obs_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.obs_horizon]
            )
            # pads by repeating the first observation
            chunk_indices = tf.maximum(chunk_indices, 0)
            traj["obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["observations"]
            )
        return traj

    def _add_goals(self, traj):
        if self.goal_conditioned:
            # NOTE: GC baselines would need next_observations/images0 -> actually no, we can just use observations/images0
            traj = GOAL_RELABELING_FUNCTIONS[self.goal_relabeling_strategy](
                traj, **self.goal_relabeling_kwargs
            )

        if self.load_language:
            lang_idx = tf.random.uniform(
                shape=[], maxval=len(traj["language"]), dtype=tf.int32
            )
            lang = traj["language"][lang_idx]
            traj["goals"]["language"] = tf.broadcast_to(
                lang, tf.shape(traj["terminals"])
            )
            traj.pop("language")

        # after goal relabeling, we can set actions and obs to chunked version
        if "action_chunks" in traj:
            traj["actions"] = traj.pop("action_chunks")
        if "obs_chunks" in traj:
            traj["observations"] = traj.pop("obs_chunks")

        return traj

    def _add_action_loss_mask(self, traj, mask_value):
        traj_len = len(traj["actions"])
        traj["action_loss_mask"] = tf.broadcast_to(tf.constant([mask_value], dtype=tf.bool), [traj_len])
        return traj

    def _augment(self, seed, image):
        if self.augment_goal_differently:
            sub_seeds = tf.unstack(
                tf.random.stateless_uniform(
                    [2, 2], seed=[seed, seed], minval=None, maxval=None, dtype=tf.int32
                )
            )
        else:
            # use the same seed for obs, and goal
            sub_seeds = [[seed, seed]] * 2

        for key, sub_seed in zip(
            ["observations", "goals"], sub_seeds
        ):
            if key not in image:
                continue
            image[key]["image"] = augment(
                image[key]["image"], sub_seed, **self.augment_kwargs
            )
        
        # NOTE: use the same seed to make sure the augmentation is consistent to observations/image
        if "random_resize_crop" in self.augment_kwargs["augment_order"]:
            image["image_flows"] = random_resized_crop(
                image["image_flows"],
                **self.augment_kwargs["random_resized_crop"],
                seed=sub_seeds[0],
            )

        return image

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

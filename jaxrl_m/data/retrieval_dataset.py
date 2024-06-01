"""
tf.data.Dataset based dataloader for the BridgeData format, meaning
TFRecords with one trajectory per example. See the BridgeDataset class
below for more details.

Written by Kevin Black (kvablack@berkeley.edu).
"""

import fnmatch
from typing import Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf
from absl import logging

from jaxrl_m.data.tf_augmentations import augment, random_resized_crop


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


class RetrievalDataset:
    """
    Args:
        data_paths: List of paths to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        batch_size: Batch size.
    """

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        batch_size: int = 256,
        act_pred_horizon: Optional[int] = None,
        prechunk: bool = False,
        include_future_obs: bool = False,
        compute_proprio_action_embedding: bool = False,
        flow_dtype: str = "float32",
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to BridgeDataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]

        # the expected type spec for the serialized examples
        self.PROTO_TYPE_SPEC = {
            "observations/images0": tf.uint8,
            "actions": tf.float32,
            "image_flows": tf.float32,
        }

        self.act_pred_horizon = act_pred_horizon
        self.prechunk = prechunk
        self.include_future_obs = include_future_obs
        self.compute_proprio_action_embedding = compute_proprio_action_embedding
        if flow_dtype == "float32":
            self.PROTO_TYPE_SPEC["image_flows"] = tf.float32
        elif flow_dtype == "float16":
            self.PROTO_TYPE_SPEC["image_flows"] = tf.float16
        else:
            raise ValueError(f"Invalid dtype: {flow_dtype}")

        self.need_proprio = (not self.prechunk) or self.compute_proprio_action_embedding
        if self.need_proprio:
            self.PROTO_TYPE_SPEC["observations/state"] = tf.float32
            self.PROTO_TYPE_SPEC["next_observations/state"] = tf.float32

        # construct a dataset for each sub-list of paths
        datasets = []
        for sub_data_paths in data_paths:
            datasets.append(self._construct_tf_dataset(sub_data_paths))

        dataset = datasets[0]
        for i in range(1, len(datasets)):
            dataset = dataset.concatenate(datasets[i])        

        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=False,
            deterministic=True,
        )

        self.tf_dataset = dataset

    def _construct_tf_dataset(self, paths: List[str]) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """

        dataset = tf.data.Dataset.from_tensor_slices(paths)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        if not self.prechunk:
            dataset = dataset.map(self._process_actions, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        if self.act_pred_horizon is not None and not self.prechunk:
            dataset = dataset.map(self._chunk_act, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        if self.include_future_obs:
            dataset = dataset.map(self._add_future_obs, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        if self.compute_proprio_action_embedding:
            dataset = dataset.map(self._compute_proprio_action_embedding, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to yield individual transitions
        # NOTE: unbatch is slow
        # dataset = dataset.unbatch()
        # NOTE: we want to keep the order here (for visualization)
        dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)

        return dataset

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
                **({"proprio": parsed_tensors["observations/state"]} if self.need_proprio else {}),
            },
            "next_observations": {
                **({"proprio": parsed_tensors["next_observations/state"]} if self.need_proprio else {}),
            },
            "actions": parsed_tensors["actions"],
            "image_flows": tf.cast(parsed_tensors["image_flows"], tf.float32),
        }

    def _process_actions(self, traj):
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
        return traj

    def _chunk_act(self, traj):
        traj_len = len(traj["actions"])
        chunk_indices = tf.broadcast_to(
            tf.range(self.act_pred_horizon), [traj_len, self.act_pred_horizon]
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None], [traj_len, self.act_pred_horizon]
        )
        # pads by repeating the last action
        chunk_indices = tf.minimum(chunk_indices, traj_len - 1)
        traj["action_chunks"] = tf.gather(traj["actions"], chunk_indices)
        traj["actions"] = traj.pop("action_chunks")
        return traj

    def _add_future_obs(self, traj):
        traj_len = len(traj["actions"])
        future_obs_indices = tf.range(traj_len) + self.act_pred_horizon
        future_obs_indices = tf.minimum(future_obs_indices, traj_len - 1)
        traj["next_observations"] = {
            "image": tf.gather(traj["observations"]["image"], future_obs_indices),
        }
        return traj
    
    def _compute_proprio_action_embedding(self, traj):
        traj_len = len(traj["actions"])
        future_obs_indices = tf.range(traj_len) + self.act_pred_horizon
        future_obs_indices = tf.minimum(future_obs_indices, traj_len - 1)
        future_proprio = tf.gather(traj["observations"]["proprio"], future_obs_indices)
        traj["proprio-action_emb"] = tf.concat([
            traj["observations"]["proprio"],
            future_proprio[..., :3] - traj["observations"]["proprio"][..., :3]
        ], axis=1)
        return traj

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

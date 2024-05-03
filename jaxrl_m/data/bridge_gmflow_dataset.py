"""
tf.data.Dataset based dataloader for the BridgeData format, meaning
TFRecords with one trajectory per example. See the BridgeDataset class
below for more details.

Written by Kevin Black (kvablack@berkeley.edu).
"""

from typing import Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf
from absl import logging


class BridgeGMFlowDataset:
    """
    Args:
        data_paths: List of paths to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        seed: Random seed.
        batch_size: Batch size.
        shuffle_buffer_size: Size of the shuffle buffer. It is split between
            sub-datasets by `sample_weights`.
        cache: Whether to cache the dataset in memory.
    """

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        batch_size: int = 256,
        act_pred_horizon: int = 1,
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to BridgeDataset: %s", kwargs)

        self.act_pred_horizon = act_pred_horizon

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

        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._add_future_obs, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to yield individual transitions
        # dataset = dataset.unbatch()
        # NOTE: use flat_map instead of interleave here because we don't want to mess up the data ordering
        dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)

        return dataset

    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/state": tf.float32,
        "actions": tf.float32,
        "terminals": tf.bool,
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
                "proprio": parsed_tensors["observations/state"],
            },
            "actions": parsed_tensors["actions"],
            "terminals": parsed_tensors["terminals"],
        }

    def _add_future_obs(self, traj):
        traj_len = len(traj["actions"])
        future_image_indices = tf.minimum(tf.range(traj_len) + self.act_pred_horizon, traj_len - 1)
        traj["future_image"] = tf.gather(traj["observations"]["image"], future_image_indices)
        return traj

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

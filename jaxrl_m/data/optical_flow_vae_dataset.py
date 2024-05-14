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

from jaxrl_m.data.tf_augmentations import random_resized_crop
from jaxrl_m.data.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS


class OpticalFlowVAEDataset:
    """
    Args:
        data_paths: List of paths to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        seed: Random seed.
        sample_weights: If data_paths is a list of list of paths, this is a
            list of weights with which to sample from each sub-list.
        batch_size: Batch size.
        shuffle_buffer_size: Size of the shuffle buffer. It is split between
            sub-datasets by `sample_weights`.
        cache: Whether to cache the dataset in memory.
        train: Whether this dataset is intended for training
            (if set to `False`, will disable shuffling and augmentations).
        augment: Whether to apply image augmentations.
        augment_kwargs: Keyword arguments for image augmentations. See
            `jaxrl_m.data.tf_augmentations.augment` for more details.
        augment_next_obs_goal_differently: Whether to use different random seeds
            for augmenting the obs, next_obs, and goal image.
    """

    PROTO_TYPE_SPEC = {}

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 10000,
        train: bool = True,
        augment: bool = False,
        augment_kwargs: dict = {},
        dtype: str = "float32",
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to BridgeDataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.augment_kwargs = augment_kwargs
        self.is_train = train
        if dtype == "float32":
            self.PROTO_TYPE_SPEC["image_flows"] = tf.float32
        elif dtype == "float16":
            self.PROTO_TYPE_SPEC["image_flows"] = tf.float16
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        # construct a dataset for each sub-list of paths
        datasets = []
        for sub_data_paths in data_paths:
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed))

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

        if train and augment:
            # apply augmentations, using a sequence of integers as seeds.
            # this was the only way I found to avoid a memory leak in tf.random.Generator
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        # NOTE: this is just a hack for viper_x data since the validation data is too small
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

    def _construct_tf_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
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

        # unbatch to yield individual transitions
        # NOTE: unbatch is slow
        # dataset = dataset.unbatch()
        # dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
        dataset = dataset.interleave(tf.data.Dataset.from_tensor_slices, num_parallel_calls=tf.data.AUTOTUNE)

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
            "image_flows": tf.cast(parsed_tensors["image_flows"], tf.float32),
        }

    def _augment(self, seed, image):

        # NOTE: only support random_resized_crop for optical flow data
        if "random_resize_crop" in self.augment_kwargs["augment_order"]:
            image["image_flows"] = random_resized_crop(
                image["image_flows"],
                **self.augment_kwargs["random_resized_crop"],
                seed=[seed, seed],
            )

        return image

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()


if __name__ == '__main__':
    dataset = OpticalFlowVAEDataset(
        ['/iliad/group/datasets/OXE_OCTO/small_test_h8_prechunk/train/out.tfrecord'],
        0,
        train=False,
        dtype="float16"
    )
    it = dataset.iterator()

    import ipdb
    ipdb.set_trace()

    tot = 0
    for batch in it:
        tot += batch['image_flows'].shape[0]
    print(tot)
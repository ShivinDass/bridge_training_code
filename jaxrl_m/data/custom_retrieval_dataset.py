"""
tf.data.Dataset based dataloader for the BridgeData format, meaning
TFRecords with one trajectory per example. See the BridgeDataset class
below for more details.

Written by Kevin Black (kvablack@berkeley.edu).
"""

from typing import List, Union
import tensorflow as tf

def safe_decode_image(image, image_size=(256, 256), dtype=tf.uint8):
    assert image.dtype == tf.string
    if tf.strings.length(image) == 0:
        return tf.zeros((*image_size, 3), dtype=dtype)
    return tf.io.decode_image(image, expand_animations=False, channels=3, dtype=dtype)

def get_dtype_from_key(key):
    if 'mask' in key:
        return tf.bool
    elif 'image' in key:
        return tf.uint8
    elif 'timestep' in key:
        return tf.int32
    elif ('dataset_name' in key) or ('language' in key):
        return tf.string
    else:
        return tf.float32

class CustomRetrievalDataset:
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
        load_keys=[],
        decode_imgs=False,
        **kwargs,
    ):
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        self.decode_imgs = decode_imgs
        print("\n==> Loading keys", load_keys, f"from {data_paths}")

        self.PROTO_TYPE_SPEC = {
            k: get_dtype_from_key(k) for k in load_keys
        }

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

        # dataset = tf.data.Dataset.from_tensor_slices(paths)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

    def _decode_example(self, example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {}
        for key, dtype in self.PROTO_TYPE_SPEC.items():
            if (key in ['observation/image_primary', 'observation/image_wrist']):
                if self.decode_imgs:
                    parsed_tensors[key] = safe_decode_image(parsed_features[key])
                else:
                    parsed_tensors[key] = parsed_features[key]
            else:
                parsed_tensors[key] = tf.io.parse_tensor(parsed_features[key], dtype)

        return {
            key: value for key, value in parsed_tensors.items()
        }

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

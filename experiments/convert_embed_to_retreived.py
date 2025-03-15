import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
import scipy
import tqdm

from jaxrl_m.data.bc_dataset import glob_to_path_list
from jaxrl_m.data.custom_retrieval_dataset import CustomRetrievalDataset

FLAGS = flags.FLAGS

flags.DEFINE_string("target_dataset_path", None, "Path to the target dataset.", required=True)
flags.DEFINE_string("prior_dataset_path", None, "Path to the prior dataset.", required=True)
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_string("output_dir", None, "Path to the output directory.", required=True)

def print_nested_dict(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print('\t' * indent, k)
            print_nested_dict(v, indent + 1)
        else:
            try:
                print('\t' * indent, k, v.shape)
            except:
                pass

def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def image_tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )

def convert_batch_to_feature(batch, i, parent_key=None):
    features = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            if k in ['observation/image_primary', 'observation/image_wrist']: # its an image
                features[k] = image_tensor_feature(v[i])
            else:
                features[k] = tensor_feature(v[i])
        else:
            raise ValueError(f"Unsupported type {type(v)}")
    return features

def main(_):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    assert os.path.exists(FLAGS.target_dataset_path), f"Path {FLAGS.target_dataset_path} does not exist."
    # load datasets
    target_paths = [[FLAGS.target_dataset_path]]
    # prior_paths  = [glob_to_path_list(FLAGS.prior_dataset_path + "/oxe_magic_soup_s?_h8_prechunk",  prefix="")]
    prior_paths  = [glob_to_path_list(FLAGS.prior_dataset_path + "/ut_datasets_h8_prechunk",  prefix="")]

    target_paths = [[os.path.join(path, "train", "out.tfrecord") for path in sub_list] for sub_list in target_paths]
    prior_paths  = [sorted([os.path.join(path, "out.tfrecord") for path in sub_list]) for sub_list in prior_paths]
    # prior_paths = [[prior_paths[0][0]]]
    # prior_paths = [[target_paths[0][0]]]
    print("Target paths:", target_paths)
    print("Prior paths:", prior_paths)

    dataset = tf.data.TFRecordDataset(target_paths[0], num_parallel_reads=tf.data.AUTOTUNE)
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        # print(example)
        all_dataset_keys = list(example.features.feature.keys())

    keys = []#all_dataset_keys
    for k in all_dataset_keys:
        if 'embedding' not in k:
            keys.append(k)

    # store retrieved data
    prior_data = CustomRetrievalDataset(
        prior_paths,
        batch_size=FLAGS.batch_size,
        load_keys=keys,
        decode_imgs=False
    )
    prior_data_iter  = prior_data.iterator()
    
    outpath_base = FLAGS.output_dir
    outpath = os.path.join(outpath_base, "out.tfrecord")
    tf.io.gfile.makedirs(os.path.dirname(outpath))
    writer = tf.io.TFRecordWriter(outpath)

    print("\nWriting retrieved data...")
    current_idx = 0
    pbar = tqdm.tqdm(total=None)
    while True:
        try:
            prior_batch = next(prior_data_iter)
            # print_nested_dict(prior_batch)
            # exit(0)
        except StopIteration:
            break

        batch_size = len(prior_batch['action'])

        for batch_idx in range(batch_size):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=convert_batch_to_feature(prior_batch, batch_idx)
                )
            )
            writer.write(example.SerializeToString())
                    
        current_idx += batch_size
        pbar.update(1)

    writer.close()

if __name__ == "__main__":
    app.run(main)
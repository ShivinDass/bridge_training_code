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

flags.DEFINE_string("dm_path", None, "Path to the target dataset.", required=True)
flags.DEFINE_string("prior_dataset_path", None, "Path to the prior dataset.", required=True)
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_string("output_dir", None, "Path to the output directory.", required=True)
flags.DEFINE_float("topk", 0.1, "Top k percentage of data to use.")

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

def load_random_mask(dm_path, topk=0.1):
    '''
    dm_path: path to the datamodels folder
    topk: top k percentage of data to use
    '''
    avg_dm = np.load(dm_path)[1000000:]
    no_nan_mask = np.logical_not(np.isnan(avg_dm))

    no_nan_idxs = np.where(no_nan_mask)[0]

    print(np.sum(no_nan_mask), len(no_nan_idxs))

    selected_idxs = np.random.permutation(no_nan_idxs)[:int(topk*len(no_nan_idxs))]
    selected_mask = np.zeros(avg_dm.shape, dtype=bool)
    selected_mask[selected_idxs] = True

    print(np.sum(selected_mask), np.sum(no_nan_mask))

    return selected_mask

    non_nan_avg_dm = avg_dm[no_nan_mask]
    print(non_nan_avg_dm.shape)
    threshold = np.sort(non_nan_avg_dm, axis=0)[int(topk*len(non_nan_avg_dm))]

    selected_dm = avg_dm <= threshold
    print(threshold, np.sum(selected_dm))

    return selected_dm

def main(_):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # load datasets
    prior_paths = []
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s1_prechunk"])
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s2_prechunk"])
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s3_prechunk"])
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s4_prechunk"])

    prior_paths  = [sorted([os.path.join(path, "out.tfrecord") for path in sub_list]) for sub_list in prior_paths]
    # prior_paths = [[prior_paths[0][0]]]
    # prior_paths = [[target_paths[0][0]]]
    print("Prior paths:", prior_paths)

    random_mask = load_random_mask(FLAGS.dm_path, topk=FLAGS.topk)

    target_dataset_name = os.path.basename(FLAGS.dm_path).split('.')[0]
    print(target_dataset_name)

    dataset = tf.data.TFRecordDataset(prior_paths[0], num_parallel_reads=tf.data.AUTOTUNE)
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        # print(example)
        all_dataset_keys = list(example.features.feature.keys())
    
    print("All dataset keys:", all_dataset_keys)

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

    outpath = os.path.join(FLAGS.output_dir, target_dataset_name + f'_random{FLAGS.topk}.tfrecord')
    # outpath = os.path.join(outpath_base, "out.tfrecord")
    tf.io.gfile.makedirs(os.path.dirname(outpath))
    writer = tf.io.TFRecordWriter(outpath)

    print("\nWriting retrieved data...")
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
            # if 'kuka' in prior_batch['dataset_name'][batch_idx].decode('utf-8'):
            #     print('kuka', random_mask[prior_batch['index'][batch_idx]])
            if random_mask[prior_batch['index'][batch_idx]]==True:
                # print('yes')
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature=convert_batch_to_feature(prior_batch, batch_idx)
                    )
                )
                writer.write(example.SerializeToString())

        pbar.update(1)

    writer.close()

    
if __name__ == "__main__":
    app.run(main)
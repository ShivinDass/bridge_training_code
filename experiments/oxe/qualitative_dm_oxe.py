import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
import scipy
import tqdm
import cv2

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

def load_dm_mask(dm_path, topk=5):
    '''
    dm_path: path to the datamodels folder
    topk: top k percentage of data to use
    '''
    avg_dm = np.load(dm_path)[1000000:]
    no_nan_mask = np.logical_not(np.isnan(avg_dm))

    non_nan_avg_dm = avg_dm[no_nan_mask]
    print(non_nan_avg_dm.shape)
    top_threshold = np.sort(non_nan_avg_dm, axis=0)[int(topk)-1]
    bot_threshold = np.sort(non_nan_avg_dm, axis=0)[-int(topk)]

    top_mask = avg_dm <= top_threshold
    bot_mask = avg_dm >= bot_threshold

    print('top_idxs', np.where(top_mask))
    print('bot_idxs', np.where(bot_mask))

    return top_mask, bot_mask

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
    print("Prior paths:", prior_paths)

    top_mask, bot_mask = load_dm_mask(FLAGS.dm_path, topk=FLAGS.topk)
    print(top_mask.sum(), bot_mask.sum())

    # dataset = tf.data.TFRecordDataset(prior_paths[0], num_parallel_reads=tf.data.AUTOTUNE)
    # for raw_record in dataset.take(1):
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     # print(example)
    #     all_dataset_keys = list(example.features.feature.keys())
    
    keys = ['observation/image_primary', 'dataset_name', 'index', 'action']

    # store retrieved data
    prior_data = CustomRetrievalDataset(
        prior_paths,
        batch_size=FLAGS.batch_size,
        load_keys=keys,
        decode_imgs=False
    )
    prior_data_iter  = prior_data.iterator()

    outpath = os.path.join(FLAGS.output_dir, os.path.basename(FLAGS.dm_path)[:-4] + f'_top.tfrecord')
    # outpath = os.path.join(outpath_base, "out.tfrecord")
    tf.io.gfile.makedirs(os.path.dirname(outpath))
    top_writer = tf.io.TFRecordWriter(outpath)
    bot_writer = tf.io.TFRecordWriter(outpath.replace('_top.tfrecord', '_bot.tfrecord'))

    print("\nWriting retrieved data...")
    pbar = tqdm.tqdm(total=None)

    all_tops = []
    top_idx = []
    top_datasets = set({})

    all_bots = []
    bot_idx = []
    bot_datasets = set({})
    while True:
        try:
            prior_batch = next(prior_data_iter)
        except StopIteration:
            break

        batch_size = len(prior_batch['action'])

        for batch_idx in range(batch_size):
            if top_mask[prior_batch['index'][batch_idx]]==True:
                all_tops.append(prior_batch['observation/image_primary'][batch_idx])
                top_idx.append(prior_batch['index'][batch_idx])
                top_datasets.add(prior_batch['dataset_name'][batch_idx].decode('utf-8'))

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature=convert_batch_to_feature(prior_batch, batch_idx)
                    )
                )
                top_writer.write(example.SerializeToString())

            if bot_mask[prior_batch['index'][batch_idx]]==True:
                all_bots.append(prior_batch['observation/image_primary'][batch_idx])
                bot_idx.append(prior_batch['index'][batch_idx])
                bot_datasets.add(prior_batch['dataset_name'][batch_idx].decode('utf-8'))

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature=convert_batch_to_feature(prior_batch, batch_idx)
                    )
                )
                bot_writer.write(example.SerializeToString())

        pbar.update(1)


    np.save(os.path.join(FLAGS.output_dir, os.path.basename(FLAGS.dm_path)[:-4] + f'_top.npy'), np.array(all_tops))
    np.save(os.path.join(FLAGS.output_dir, os.path.basename(FLAGS.dm_path)[:-4] + f'_bot.npy'), np.array(all_bots))
    np.save(os.path.join(FLAGS.output_dir, os.path.basename(FLAGS.dm_path)[:-4] + f'_top_idx.npy'), np.array(top_idx))
    np.save(os.path.join(FLAGS.output_dir, os.path.basename(FLAGS.dm_path)[:-4] + f'_bot_idx.npy'), np.array(bot_idx))
    print(top_datasets, bot_datasets)
    top_writer.close()
    bot_writer.close()



    # all_tops = []
    # top_datasets = set({})

    # all_bots = []
    # bot_datasets = set({})
    # while True:
    #     try:
    #         prior_batch = next(prior_data_iter)
    #     except StopIteration:
    #         break

    #     batch_size = len(prior_batch['action'])

    #     for batch_idx in range(batch_size):
    #         if top_mask[prior_batch['index'][batch_idx]]==True:
    #             all_tops.append(prior_batch['observation/image_primary'][batch_idx])
    #             top_datasets.add(prior_batch['dataset_name'][batch_idx].decode('utf-8'))

    #         if bot_mask[prior_batch['index'][batch_idx]]==True:
    #             all_bots.append(prior_batch['observation/image_primary'][batch_idx])
    #             bot_datasets.add(prior_batch['dataset_name'][batch_idx].decode('utf-8'))
    #     pbar.update(1)

    # print()
    # print(np.array(all_tops).shape)
    # print(np.array(all_bots).shape)
    # print('top datasets:', top_datasets)
    # print('bot datasets:', bot_datasets)
    # print()

    # all_tops = np.random.permutation(all_tops)
    # all_bots = np.random.permutation(all_bots)

    # outpath = '/home/shivin/foundation_models/z_qualitative/droid'
    # top_path = os.path.join(outpath, 'top')
    # bot_path = os.path.join(outpath, 'bot')
    
    # for i in range(0, 150, 10):
    #     cv2.imwrite(os.path.join(top_path, f'{i}.png'), cv2.cvtColor(np.concatenate(all_tops[i:i+10], axis=1), cv2.COLOR_RGB2BGR))
    #     cv2.imwrite(os.path.join(bot_path, f'{i}.png'), cv2.cvtColor(np.concatenate(all_bots[i:i+10], axis=1), cv2.COLOR_RGB2BGR))

    
if __name__ == "__main__":
    app.run(main)
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
flags.DEFINE_bool("random", False, "", required=False)
flags.DEFINE_float("topk", 0.1, "Top k percentage of data to use.")
flags.DEFINE_float("iter_dm", -1, "Number of dm iterations to use.")

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

# def load_dm_mask_single_iter(dm_path, topk=0.1):
#     filtered_dw = np.load(os.path.join(dm_path, 'data_weights.npy'))[:1000000]
#     filtered_dm = np.load(os.path.join(dm_path, 'datamodels.npy'))[1000000:]
    
#     n_trajs = 4500
#     threshold = np.sort(filtered_dm, axis=0)[int(topk*n_trajs)]
#     selected_dm = filtered_dm <= threshold
#     return selected_dm

#     mask = np.logical_and(filtered_dm, filtered_dw)
#     index = np.where(mask)[0]
#     return mask

def load_dm_mask(dm_path, topk=0.1, iter_dm=-1):
    iter_num = len(os.listdir(dm_path)) if iter_dm == -1 else int(iter_dm)
    all_dms = []
    count_dms = 0

    # n_trajs = 4500 # traj
    n_trajs = 24489 # w30
    # n_trajs = 46704 # w15
    for iter_id in range(iter_num):
        iter_path = os.path.join(dm_path, f'iter_{iter_id}')
        
        try:
            # filtered_dw = np.load(os.path.join(iter_path, 'data_weights.npy'))[:1000000]
            filtered_dm = np.load(os.path.join(iter_path, 'datamodels.npy'))[1000000:1000000+n_trajs]
            all_dms.append(filtered_dm)
            count_dms += 1
        except:
            pass
    all_dms = np.array(all_dms)
    avg_dm = np.mean(all_dms, axis=0, where=all_dms!=0)
    avg_dm = np.nan_to_num(avg_dm)

    print(count_dms, np.sum(avg_dm!=0), n_trajs)
    # assert n_trajs == np.sum(avg_dm!=0)

    # Option 1
    threshold = np.sort(avg_dm, axis=0)[int(topk*n_trajs)]
    selected_dm = avg_dm <= threshold
    print(threshold, np.sum(selected_dm))

    # Option 2
    # min_th, max_th = np.percentile(avg_dm, [100*topk/2, 100*(1-topk/2)])
    # selected_dm = (avg_dm <= min_th) | (avg_dm >= max_th)
    # print(min_th, max_th, np.sum(selected_dm))

    return selected_dm, count_dms

def main(_):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # prior_paths  = [glob_to_path_list(FLAGS.prior_dataset_path + "/oxe_magic_soup_s?_h8_prechunk",  prefix="")]
    prior_paths  = [glob_to_path_list(FLAGS.prior_dataset_path,  prefix="")]
    prior_paths  = [sorted([os.path.join(path, "out.tfrecord") for path in sub_list]) for sub_list in prior_paths]
    # prior_paths = [[prior_paths[0][0]]]
    print("Prior paths:", prior_paths)
    print("Dm path:", FLAGS.dm_path)

    dm_mask, iter_dm = load_dm_mask(FLAGS.dm_path, topk=FLAGS.topk, iter_dm=FLAGS.iter_dm)
    
    # print(FLAGS.random)
    # if FLAGS.random == True:
    #     total = np.sum(dm_mask)
    #     random_indices = np.random.permutation(4500)[:total]
    #     dm_mask = np.zeros(4500, dtype=bool)
    #     dm_mask[random_indices] = True
    # print(np.sum(dm_mask))
    # print(dm_mask.shape)

    target_dataset_name = os.path.basename(FLAGS.dm_path)
    print(target_dataset_name)

    dataset = tf.data.TFRecordDataset(prior_paths[0], num_parallel_reads=tf.data.AUTOTUNE)
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

    outpath = os.path.join(FLAGS.output_dir, target_dataset_name + f'_iter{iter_dm}_top{FLAGS.topk}.tfrecord')
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
            if dm_mask[prior_batch['index'][batch_idx][0]]==True:
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
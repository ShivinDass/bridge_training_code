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
flags.DEFINE_float("threshold", 0.1, "Threshold for retrieval.")
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

# RETRIEVAL_METHODS = ['language'] 
RETRIEVAL_METHODS = ['br', 'flow', 'action']

def get_embedding_from_batch(batch, method):
    if method == 'br':
        return batch['br_embedding']
    elif method == 'flow':
        return batch['flow_embedding']
    elif method == 'action':
        return batch['action'].reshape(batch['action'].shape[0], -1)
    elif method == 'language':
        return np.zeros((batch['action'].shape[0], 2))
    else:
        raise ValueError(f"Invalid retrieval method: {method}")

def main(_):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    assert os.path.exists(FLAGS.target_dataset_path), f"Path {FLAGS.target_dataset_path} does not exist."
    # load datasets
    target_paths = [[FLAGS.target_dataset_path]]
    prior_paths = []
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s1_prechunk"])
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s2_prechunk"])
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s3_prechunk"])
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s4_prechunk"])
    
    target_paths = [[os.path.join(path, "out.tfrecord") for path in sub_list] for sub_list in target_paths]
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

    target_data = CustomRetrievalDataset(
        target_paths,
        batch_size=FLAGS.batch_size,
        load_keys=['br_embedding', 'flow_embedding', 'action', 'task/language_instruction'],
        decode_imgs=False
    )
    target_data_iter = target_data.iterator()
    target_batch = next(target_data_iter)

    target_embeddings = {method: [] for method in RETRIEVAL_METHODS}
    while True:
        for method in RETRIEVAL_METHODS:
            target_embeddings[method].append(get_embedding_from_batch(target_batch, method))

        try:
            target_batch = next(target_data_iter)
        except StopIteration:
            break
    
    for method in RETRIEVAL_METHODS:
        target_embeddings[method] = jnp.concatenate(target_embeddings[method], axis=0)
        print(f"{method} target shape:", target_embeddings[method].shape)
    
    logging.info("Finish computing target embeddings.")

    # compute prior embeddings
    prior_data = CustomRetrievalDataset(
        prior_paths,
        batch_size=FLAGS.batch_size,
        load_keys=['br_embedding', 'flow_embedding', 'action', 'task/language_instruction', 'index'],
        decode_imgs=False
    )
    prior_data_iter  = prior_data.iterator()
    sim_scores = {method: [] for method in RETRIEVAL_METHODS}
    pbar = tqdm.tqdm(total=None)
    while True:
        pbar.update(1)
        try:
            prior_batch = next(prior_data_iter)
        except StopIteration:
            break

        for method in RETRIEVAL_METHODS:
            if method == 'language':
                prior_language = prior_batch['task/language_instruction']
                required_langauge = ["turn on the_stove",
                                     "put the moka pot on the stove",]
                                    #  "turn on the stove and put the frying pan on it"]
                
                sim_score = []
                for lang in prior_language:
                    lang = lang.decode('utf-8')
                    if lang in required_langauge:
                        sim_score.append(1)
                    else:
                        sim_score.append(0)
                
                sim_score = np.array(sim_score)
                sim_scores[method].append(sim_score)
            else:
                prior_embeddings = get_embedding_from_batch(prior_batch, method)
                sim_scores[method].append(-jnp.min(scipy.spatial.distance.cdist(target_embeddings[method], prior_embeddings), axis=0))

    for method in RETRIEVAL_METHODS:
        sim_scores[method] = jnp.concatenate(sim_scores[method], axis=0)
        logging.info(f"prior size {method}: {sim_scores[method].shape[0]}")
    logging.info("Finish computing similarity scores.")

    if len(RETRIEVAL_METHODS) == 1 and RETRIEVAL_METHODS[0] == 'language':
        FLAGS.threshold = np.sum(sim_scores['language'])
        print(FLAGS.threshold)

    # find retrieved data
    masks = {}
    for method in RETRIEVAL_METHODS:
        retrieval_distances = -sim_scores[method]
        sorted_idx = np.argsort(retrieval_distances)
        if FLAGS.threshold > 1:
            threshold_idx = sorted_idx[:int(FLAGS.threshold)]
        else:
            threshold_idx = sorted_idx[:int(FLAGS.threshold * len(sorted_idx))]
        mask = np.zeros_like(retrieval_distances, dtype=np.bool_)
        mask[threshold_idx] = True

        masks[method] = mask.copy()
        logging.info(f"selected count {method}: {np.sum(masks[method])}")

    keys = []#all_dataset_keys
    for k in all_dataset_keys:
        if 'embedding' not in k and 'is_suboptimal' not in k:
            keys.append(k)

    # store retrieved data
    prior_data = CustomRetrievalDataset(
        prior_paths,
        batch_size=FLAGS.batch_size,
        load_keys=keys,
        decode_imgs=False
    )
    prior_data_iter  = prior_data.iterator()

    target_dataset_name = os.path.basename(FLAGS.target_dataset_path)
    outpath_base = os.path.join(FLAGS.output_dir, f"{target_dataset_name}_th{FLAGS.threshold}")
    writers = {}
    for method in RETRIEVAL_METHODS:
        outpath = os.path.join(outpath_base, method, "out.tfrecord")
        tf.io.gfile.makedirs(os.path.dirname(outpath))
        writers[method] = tf.io.TFRecordWriter(outpath)

        # move val data to corresponding directory
        # os.system(f"cp -r {FLAGS.target_dataset_path}/val {os.path.join(outpath_base, method)}")
    
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
        for method in RETRIEVAL_METHODS:
            current_mask = masks[method][current_idx:current_idx+batch_size]

            for batch_idx in range(batch_size):
                if current_mask[batch_idx]:
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=convert_batch_to_feature(prior_batch, batch_idx)
                        )
                    )
                    writers[method].write(example.SerializeToString())
                    
        current_idx += batch_size
        pbar.update(1)

    for method in RETRIEVAL_METHODS:
        writers[method].close()

    
if __name__ == "__main__":
    app.run(main)
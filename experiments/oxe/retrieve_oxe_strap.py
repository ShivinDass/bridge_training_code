import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
import scipy
import tqdm

from jaxrl_m.data.bc_dataset import glob_to_path_list
from jaxrl_m.utils.strap_utils import slice_embeddings, get_single_match
from jaxrl_m.data.custom_retrieval_dataset import CustomRetrievalDataset

FLAGS = flags.FLAGS

flags.DEFINE_string("target_dataset_path", None, "Path to the target dataset.", required=True)
flags.DEFINE_string("prior_dataset_path", None, "Path to the prior dataset.", required=True)
flags.DEFINE_float("threshold", 0.1, "Threshold for retrieval.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_string("output_dir", None, "Path to the output directory.", required=True)

def write_video(video_frames, filename, fps=10):
    '''
    video_frames: list of frames (T, H, W, C)
    '''

    import imageio
    # for i in range(len(video_frames)):
    #     video_frames[i] = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
    imageio.mimwrite(filename, video_frames, fps=fps)


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

def convert_batch_to_feature(batch):
    features = {}
    for k, v in batch.items():
        # if isinstance(v, np.ndarray):
        if k in ['observation/image_primary', 'observation/image_wrist']: # its an image
            features[k] = image_tensor_feature(v)
        else:
            features[k] = tensor_feature(v)
        # else:
        #     raise ValueError(f"{k}, Unsupported type {type(v)}")
    return features

def single_sample_iterator_wrapper(iterator):
    pbar = tqdm.tqdm(total=None)
    while True:
        try:
            batch = next(iterator)
            for i in range(len(batch['action'])):
                yield {k: v[i] for k, v in batch.items()}
        except StopIteration:
            break
        pbar.update(1)
    
def stack_key(traj, key):
    return np.stack([t[key] for t in traj])

def trajectory_iterator(dataset: CustomRetrievalDataset):
    target_data_iter = single_sample_iterator_wrapper(dataset.iterator())
    
    prev_index = None
    traj = []
    for raw_record in target_data_iter:
        cur_index = np.squeeze(raw_record['index'])
        if prev_index != cur_index:
            if prev_index is not None:
                yield traj
            traj = []
        raw_record['timestep'] = np.array([len(traj)])
        traj.append(raw_record)
        prev_index = cur_index

def main(_):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    assert os.path.exists(FLAGS.target_dataset_path), f"Path {FLAGS.target_dataset_path} does not exist."
    # load datasets
    target_paths = [[FLAGS.target_dataset_path]]
    prior_paths = []

    target_dataset_name = os.path.basename(FLAGS.target_dataset_path)
    outpath_base = os.path.join(FLAGS.output_dir, f"{target_dataset_name}_th{FLAGS.threshold}")
    outpath = os.path.join(outpath_base, 'strap', "out.tfrecord")
    tf.io.gfile.makedirs(os.path.dirname(outpath))

    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s1_prechunk"])
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s2_prechunk"])
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s3_prechunk"])
    prior_paths.append([FLAGS.prior_dataset_path + "/oxe_magic_soup_s4_prechunk"])
    
    target_paths = [[os.path.join(path, "out.tfrecord") for path in sub_list] for sub_list in target_paths]
    prior_paths  = [sorted([os.path.join(path, "out.tfrecord") for path in sub_list]) for sub_list in prior_paths]
    print("Target paths:", target_paths)
    print("Prior paths:", prior_paths)

    dataset = tf.data.TFRecordDataset(target_paths[0], num_parallel_reads=tf.data.AUTOTUNE)
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        # print(example)
        all_dataset_keys = list(example.features.feature.keys())
    print(all_dataset_keys)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    target_data = CustomRetrievalDataset(
        target_paths,
        batch_size=FLAGS.batch_size,
        load_keys=['dino_embeddings', 'observation/proprio', 'action', 'index'],
        # load_keys=['br_embedding', 'flow_embedding', 'action', 'index', 'task/language_instruction', 'observation/image_primary'], #
        decode_imgs=False,
        num_parallel_calls=1,
        num_parallel_reads=1
    )
    
    eef_poses = []
    task_embeddings = []
    for traj in trajectory_iterator(target_data):
        dino_embedding = stack_key(traj, 'dino_embeddings')
        eef_pose = stack_key(traj, 'observation/proprio')

        task_embeddings.append(dino_embedding)
        eef_poses.append(eef_pose)
    
    task_embeddings = slice_embeddings(eef_poses=eef_poses, task_embeddings=task_embeddings, min_length=20)
    logging.info(f"Finish computing target embeddings. # embeddings = {len(task_embeddings)}")
    # print([len(task_emb) for task_emb in task_embeddings])
    # exit(0)

    # compute prior embeddings
    prior_data = CustomRetrievalDataset(
        prior_paths,
        batch_size=FLAGS.batch_size,
        load_keys=['dino_embeddings', 'action', 'index'],
        decode_imgs=False,
        num_parallel_calls=1,
        num_parallel_reads=1
    )

    total_prior_steps = 0
    per_task_emb_subseq = [[] for _ in range(len(task_embeddings))]
    for traj in trajectory_iterator(prior_data):
        dino_embedding = stack_key(traj, 'dino_embeddings')
        traj_index = traj[0]['index']
        total_prior_steps += len(dino_embedding)

        for i, task_emb in enumerate(task_embeddings):
            subseq = get_single_match(task_emb, dino_embedding, traj_index)
            if subseq is None:
                continue
            per_task_emb_subseq[i].append(subseq)

    for i, _ in enumerate(per_task_emb_subseq):
        per_task_emb_subseq[i].sort()
    print("Finish computing prior embeddings.")

    to_retrieve = {}
    retrieved_steps = 0
    level = 0
    while True:
        for i, task_emb_subseq in enumerate(per_task_emb_subseq):
            if len(task_emb_subseq) > level:
                if task_emb_subseq[level].traj_index in to_retrieve:
                    to_retrieve[task_emb_subseq[level].traj_index].append(task_emb_subseq[level])
                else:
                    to_retrieve[task_emb_subseq[level].traj_index] = [task_emb_subseq[level]]
                retrieved_steps += len(task_emb_subseq[level])
        
        if retrieved_steps >= total_prior_steps * FLAGS.threshold:
            break
        level += 1

    print(retrieved_steps, total_prior_steps, len(to_retrieve), level)

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

    writer = tf.io.TFRecordWriter(outpath)

    print("\nWriting retrieved data...")
    for traj in trajectory_iterator(prior_data):
        traj_index = traj[0]['index']
        if traj_index not in to_retrieve:
            continue
        for subseq in to_retrieve[traj_index]:
            for i in range(subseq.start, subseq.end):
                example = tf.train.Example(
                        features=tf.train.Features(
                            feature=convert_batch_to_feature(traj[i])
                        )
                    )
                writer.write(example.SerializeToString())
                
    writer.close()
    
if __name__ == "__main__":
    app.run(main)
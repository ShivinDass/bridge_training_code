import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags
import scipy

from jaxrl_m.agents import pretrain_agents
from jaxrl_m.data.bridge_dataset import glob_to_path_list
from jaxrl_m.data.bridge_retrieval_dataset import BridgeRetrievalDataset
from jaxrl_m.vision import encoders, decoders
from jaxrl_m.data.text_processing import text_processors

FLAGS = flags.FLAGS

flags.DEFINE_string("prior_dataset_path", None, "Path to the prior dataset.", required=True)
flags.DEFINE_string("precomputed_sim_scores_path", None, "Path to precomputed sim scores", required=True)
flags.DEFINE_float("threshold", 0.1, "Threshold for retrieval.")
flags.DEFINE_string("output_dir", None, "Path to the output directory.", required=True)
flags.DEFINE_string("postfix", "", "Postfix for the output path.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def main(_):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # load datasets
    prior_paths  = [glob_to_path_list(FLAGS.prior_dataset_path,  prefix=FLAGS.config.data_path)]
    prior_paths  = [sorted([os.path.join(path, "train/out.tfrecord") for path in sub_list]) for sub_list in prior_paths]
    prior_data = BridgeRetrievalDataset(
        prior_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
    )

    sim_scores = np.load(FLAGS.precomputed_sim_scores_path)
    retrieval_distances = -sim_scores
    sorted_distances = np.argsort(retrieval_distances)
    threshold_idx = sorted_distances[:int(FLAGS.threshold * len(sorted_distances))]
    mask = np.zeros_like(retrieval_distances, dtype=np.bool_)
    mask[threshold_idx] = True

    outpath = os.path.join(FLAGS.output_dir, f"{FLAGS.prior_dataset_path.split('/')[0]}_{FLAGS.postfix}_{FLAGS.threshold}", 'train/out.tfrecord')
    tf.io.gfile.makedirs(os.path.dirname(outpath))
    with tf.io.TFRecordWriter(outpath) as writer:
        prior_data_iter  = prior_data.tf_dataset.as_numpy_iterator()
        current_idx, logger_step = 0, 0

        while True:
            try:
                prior_batch = next(prior_data_iter)
                if logger_step == 0:
                    logging.info(f"First three actions of the first batch: {prior_batch['actions'][:3]}")
                current_mask = mask[current_idx:current_idx+len(prior_batch['terminals'])]
                current_idx += len(prior_batch['terminals'])
                logger_step += 1
            except StopIteration:
                break

            if np.sum(current_mask) != 0:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "observations/images0": tensor_feature(
                                prior_batch["observations"]["image"][current_mask]
                            ),
                            "observations/state": tensor_feature(
                                prior_batch["observations"]["proprio"][current_mask]
                            ),
                            "next_observations/images0": tensor_feature(
                                prior_batch["next_observations"]["image"][current_mask]
                            ),
                            "next_observations/state": tensor_feature(
                                prior_batch["next_observations"]["proprio"][current_mask]
                            ),
                            "actions": tensor_feature(
                                prior_batch["actions"][current_mask]
                            ),
                            "terminals": tensor_feature(
                                prior_batch["terminals"][current_mask]
                            ),
                            "truncates": tensor_feature(prior_batch["truncates"][current_mask]),
                            "image_flows": tensor_feature(
                                prior_batch["image_flows"][current_mask]
                            ),
                        }
                    )
                )
                writer.write(example.SerializeToString())

            if logger_step % 100 == 0:
                logging.info(f"Processed {logger_step} batches.")


if __name__ == "__main__":
    app.run(main)
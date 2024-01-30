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
flags.DEFINE_string("output_dir", None, "Path to the output directory.", required=True)
flags.DEFINE_list("key_words", None, "Key words for retrieval.", required=True)

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
    logging.info(f"Using key words: {FLAGS.key_words}")

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # load datasets
    prior_paths  = [glob_to_path_list(FLAGS.prior_dataset_path,  prefix=FLAGS.config.data_path)]
    prior_paths  = [sorted([os.path.join(path, "train/out.tfrecord") for path in sub_list]) for sub_list in prior_paths]
    prior_data = BridgeRetrievalDataset(
        prior_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        load_language=True,
    )

    tot = 0
    outpath = os.path.join(FLAGS.output_dir, f"{FLAGS.prior_dataset_path.split('/')[0]}_{'-'.join(FLAGS.key_words)}", 'train/out.tfrecord')
    tf.io.gfile.makedirs(os.path.dirname(outpath))
    with tf.io.TFRecordWriter(outpath) as writer:
        prior_data_iter  = prior_data.tf_dataset.as_numpy_iterator()
        logger_step = 0

        while True:
            try:
                prior_batch = next(prior_data_iter)
                current_mask = [False] * len(prior_batch['terminals'])
                for i in range(len(prior_batch['terminals'])):
                    for key_word in FLAGS.key_words:
                        if key_word in prior_batch['goal_language'][i].decode("utf-8"):
                            current_mask[i] = True
                            break
                current_mask = np.array(current_mask)
                tot += np.sum(current_mask)
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
    
    logging.info(f"Total number of retrieved samples: {tot}")


if __name__ == "__main__":
    app.run(main)
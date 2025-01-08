import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags

from jaxrl_m.agents import pretrain_agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger
import wandb
from jaxrl_m.data.bc_dataset import glob_to_path_list
from jaxrl_m.data.optical_flow_vae_dataset import ImageActionVAEDataset
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders, decoders
from jaxrl_m.data.text_processing import text_processors
from tqdm import tqdm
try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "data_config",
    None,
    "File path to the data configuration.",
    lock_config=False,
)

def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({"project": "baselines_oxe", "exp_descriptor": FLAGS.name})
    wandb_logger = WandBLogger(
        wandb_config=wandb_config, variant=FLAGS.config.to_dict(), debug=FLAGS.debug
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # load datasets
    assert type(FLAGS.data_config.include[0]) == list
    task_paths = [
        glob_to_path_list(
            path, prefix=FLAGS.config.data_path, exclude=FLAGS.data_config.exclude
        )
        for path in FLAGS.data_config.include
    ]

    train_paths = [
        [os.path.join(path, "train/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]
    val_paths = [
        [os.path.join(path, "val/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]

    train_data = ImageActionVAEDataset(
        train_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        train=True,
        sample_weights=FLAGS.data_config.sample_weights,
        dtype=FLAGS.data_config.dtype,
        **FLAGS.config.dataset_kwargs,
    )
    val_data = ImageActionVAEDataset(
        val_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        train=False,
        dtype=FLAGS.data_config.dtype,
        **FLAGS.config.dataset_kwargs,
    )

    train_data_iter = map(
        shard_fn, train_data.tf_dataset.as_numpy_iterator()
    )

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['image'].shape[0] // num_devices}"
    )
    print("Example batch shape: ", example_batch['image'].shape, example_batch['actions'].shape)
    # define encoder
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # define decoder
    decoder_def = decoders[FLAGS.config.decoder](**FLAGS.config.decoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = pretrain_agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch,
        encoder=encoder_def,
        decoder=decoder_def,
        **FLAGS.config.agent_kwargs,
    )
    if FLAGS.config.resume_path is not None:
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    for i in tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        batch = next(train_data_iter)
        timer.tock("dataset")

        timer.tick("train")
        agent, update_info = agent.update(batch)
        timer.tock("train")

        if (i + 1) % 5000 == 0:
            logging.info("Visualizing reconstructions...")
            rng, val_rng = jax.random.split(rng)
            recon_images = agent.visualize_reconstruction(batch, seed=val_rng)

            original_images = tf.cast(255*batch["image"][:10], tf.uint8).numpy()
            original_images = np.concatenate(original_images, axis=1)

            recon_images = tf.cast(255*recon_images[:10], tf.uint8).numpy()
            recon_images = np.concatenate(recon_images, axis=1)

            # original_images = normalize_image_list(tf.cast(batch["image"][:10], tf.float32).numpy())
            # second_channel = np.zeros_like(original_images[..., 0:1])
            # original_images = np.concatenate([original_images, second_channel], axis=-1)
            # original_images = np.concatenate(original_images, axis=1)

            # recon_images = normalize_image_list(tf.cast(recon_images[:10], tf.float32).numpy())
            # second_channel = np.zeros_like(recon_images[..., 0:1])
            # recon_images = np.concatenate([recon_images, second_channel], axis=-1)
            # recon_images = np.concatenate(recon_images, axis=1)

            images = np.concatenate([original_images, recon_images], axis=0)
            wandb_logger.log({"reconstructions": wandb.Image(images)}, step=i)

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")

            timer.tick("val")
            metrics = []
            val_data_iter = map(shard_fn, val_data.iterator())
            for batch in val_data_iter:
                rng, val_rng = jax.random.split(rng)
                metrics.append(agent.get_debug_metrics(batch, seed=val_rng))
            if len(metrics) > 1:
                metrics = metrics[:-1] # drop remainder
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_logger.log({"validation": metrics}, step=i)
            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=i + 1, keep=1e6
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log({"training": update_info}, step=i)

            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


if __name__ == "__main__":
    app.run(main)

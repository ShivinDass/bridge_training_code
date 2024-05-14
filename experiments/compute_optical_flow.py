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
import torch
import torch.nn.functional as F

from jaxrl_m.data.bc_dataset import glob_to_path_list
from jaxrl_m.data.bridge_gmflow_dataset import BridgeGMFlowDataset
from jaxrl_m.utils.timer_utils import Timer
from gmflow.gmflow.gmflow import GMFlow

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("horizon", None, "Horizon to compute the optical flow", required=True)
flags.DEFINE_string("data_dir", None, "Path to the data directory.", required=True)
flags.DEFINE_string("out_dir_prefix", None, "Path to the output directory.", required=True)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def create_optical_flow_tfrecord(model, device, inference_size, outpath, iter):
    tf.io.gfile.makedirs(os.path.dirname(outpath))
    with tf.io.TFRecordWriter(outpath) as writer:
        while True:
            try:
                batch = next(iter)
                image = batch['observations']['image'] # (B, H, W, C)
                future_image = batch['future_image'] # (B, H , W, C)
                image = torch.from_numpy(image.copy()).permute(0, 3, 1, 2).to(device).float() # (B, C, H, W)
                future_image = torch.from_numpy(future_image.copy()).permute(0, 3, 1, 2).to(device).float() # (B, C, H, W)
                
                ori_size = image.shape[-2:]
                image = F.interpolate(image, size=inference_size, mode='bilinear', align_corners=True)
                future_image = F.interpolate(future_image, size=inference_size, mode='bilinear', align_corners=True)            

                with torch.no_grad():
                    results_dict = model(image, future_image,
                                        attn_splits_list=[2],
                                        corr_radius_list=[-1],
                                        prop_radius_list=[-1],
                                        pred_bidir_flow=False,
                                        )

                flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
                image_flows = flow_pr.permute(0, 2, 3, 1).cpu().numpy() # (B, H, W, 2)

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "observations/images0": tensor_feature(batch["observations"]["image"]),
                            "observations/state": tensor_feature(batch["observations"]["proprio"]),
                            "actions": tensor_feature(batch["actions"]),
                            "terminals": tensor_feature(batch["terminals"]),
                            "image_flows": tensor_feature(image_flows),
                        }
                    )
                )
                writer.write(example.SerializeToString())

            except StopIteration:
                break


def main(_):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # load datasets
    assert type(FLAGS.bridgedata_config.include[0]) == list
    task_paths = [
        glob_to_path_list(
            path, prefix=FLAGS.data_dir, exclude=FLAGS.bridgedata_config.exclude
        )
        for path in FLAGS.bridgedata_config.include
    ]

    train_paths = [
        [os.path.join(path, "train/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]
    possible_val_paths = [
        [os.path.join(path, "val/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]
    val_paths = []
    for val_path in possible_val_paths:
        if os.path.exists(val_path[0]):
            val_paths.append(val_path)

    train_data = BridgeGMFlowDataset(
        train_paths,
        batch_size=FLAGS.batch_size,
        act_pred_horizon=FLAGS.horizon,
    )
    val_data = BridgeGMFlowDataset(
        val_paths,
        batch_size=FLAGS.batch_size,
        act_pred_horizon=FLAGS.horizon,
    )

    train_data_iter = train_data.tf_dataset.as_numpy_iterator()
    val_data_iter = val_data.tf_dataset.as_numpy_iterator()

    # load GMFLow model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GMFlow(feature_channels=128,
                    num_scales=1,
                    upsample_factor=8,
                    num_head=1,
                    attention_type='swin',
                    ffn_dim_expansion=4,
                    num_transformer_layers=6,
                    ).to(device)
    checkpoint = torch.load('gmflow/pretrained/gmflow_sintel-0c07dcb3.pth')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights)
    model.eval()
    inference_size = [480, 480] # Use larger inference size for better quality

    train_outpath = os.path.join(FLAGS.data_dir, f"{FLAGS.out_dir_prefix}_h{FLAGS.horizon}_prechunk", "train/out.tfrecord")
    create_optical_flow_tfrecord(model, device, inference_size, train_outpath, train_data_iter)
    val_outpath = os.path.join(FLAGS.data_dir, f"{FLAGS.out_dir_prefix}_h{FLAGS.horizon}_prechunk", "val/out.tfrecord")
    create_optical_flow_tfrecord(model, device, inference_size, val_outpath, val_data_iter)


if __name__ == "__main__":
    app.run(main)

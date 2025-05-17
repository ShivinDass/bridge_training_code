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
import tqdm

from jaxrl_m.data.bc_dataset import glob_to_path_list
from jaxrl_m.data.embed_dataset import EmbedDataset
from jaxrl_m.utils.timer_utils import Timer
from gmflow.gmflow.gmflow import GMFlow

from jaxrl_m.vision import encoders, decoders
from jaxrl_m.agents import pretrain_agents
from experiments.configs.pretrain_config import get_config

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

dino_model = None
gm_flow_model = None
flow_embed_model = None
br_embed_model = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

WINDOW_SIZE = 1
FLAGS = flags.FLAGS

flags.DEFINE_string("data_name", None, "Data name to use.", required=True)
flags.DEFINE_string("data_dir", None, "Path to the data directory.", required=True)
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("future_image_horizon", None, "Horizon to compute the optical flow", required=True)
flags.DEFINE_string("out_dir", None, "Path to the output directory.", required=True)
flags.DEFINE_string("data_stats_path", None, "Path to the dataset statistics.")
flags.DEFINE_integer("act_pred_horizon", None, "action prediction horizon", required=True)


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

# def image_tensor_feature(value):
#     value = tf.convert_to_tensor(value)
#     value = tf.cast(value, tf.uint8)
#     return tf.train.Feature(
#         bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
#     )
def image_tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )

def convert_batch_to_feature(batch, i, parent_key=None):
    features = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            converted_dict = convert_batch_to_feature(v, i, parent_key=k)
            for key, value in converted_dict.items():
                features[f"{k}/{key}"] = value
        elif isinstance(v, np.ndarray):
            if k in ['image_primary', 'image_wrist'] and parent_key=='observation': # its an image
                features[k] = image_tensor_feature(v[i][0])
            else:
                if parent_key=='observation':
                    features[k] = tensor_feature(v[i][0])
                else:
                    features[k] = tensor_feature(v[i])
        else:
            raise ValueError(f"Unsupported type {type(v)}")
    return features

def compute_flow_embeddings(batch):
    image = batch['observation']['image_primary'][:, 0] # (B, H, W, C)
    future_image = batch['future_image'] # (B, H , W, C)
    image = torch.from_numpy(image.copy()).permute(0, 3, 1, 2).to(DEVICE).float() # (B, C, H, W)
    future_image = torch.from_numpy(future_image.copy()).permute(0, 3, 1, 2).to(DEVICE).float() # (B, C, H, W)
    
    image = F.interpolate(image, size=(128, 128), mode='bilinear', align_corners=True)
    future_image = F.interpolate(future_image, size=(128, 128), mode='bilinear', align_corners=True)          

    with torch.no_grad():
        results_dict = gm_flow_model(image, future_image,
                            attn_splits_list=[2],
                            corr_radius_list=[-1],
                            prop_radius_list=[-1],
                            pred_bidir_flow=False,
                            )

    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
    image_flows = flow_pr.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16) # (B, H, W, 2)

    new_batch = {"image_flows": image_flows}
    if flow_embed_model is None:
        load_flow_embed_model(new_batch)

    # compute flow embeddings
    embeddings = flow_embed_model.compute_embeddings(new_batch)
    
    # embed_flow
    return embeddings

def compute_br_embeddings(batch):
    new_batch = {
        'image': batch['observation']['image_primary'][:, 0],
        'actions': batch['action'],
    }

    if br_embed_model is None:
        load_br_embed_model(new_batch)
    
    # compute br embeddings
    embeddings = br_embed_model.compute_embeddings(new_batch)
    return embeddings

def compute_dino_embeddings(batch):
    if dino_model is None:
        load_dino_model()
    # preprocess images
    images = batch['observation']['image_primary'][:, 0] # (B, H, W, C)
    images = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to(DEVICE).float() # (B, C, H, W)

    with torch.no_grad():
        features = dino_model.preprocess(images)
        features = dino_model.encode(images)

    return features.cpu().numpy()

def load_gm_flow_model():
    # load GMFLow model
    global gm_flow_model
    gm_flow_model = GMFlow(feature_channels=128,
                    num_scales=1,
                    upsample_factor=8,
                    num_head=1,
                    attention_type='swin',
                    ffn_dim_expansion=4,
                    num_transformer_layers=6,
                    ).to(DEVICE)
    checkpoint = torch.load('gmflow/pretrained/gmflow_sintel-0c07dcb3.pth')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    gm_flow_model.load_state_dict(weights)
    gm_flow_model.eval()

def load_dino_model():
    from jaxrl_m.utils.strap_utils import DinoV2
    global dino_model
    dino_model = DinoV2(device=DEVICE)
    
def load_flow_embed_model(target_batch):
    # load flow_embed model
    global flow_embed_model
    config = get_config("optical_flow_vae")

    # define encoder and decoder
    encoder_def = encoders[config.encoder](**config.encoder_kwargs)
    decoder_def = decoders[config.decoder](**config.decoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(config.seed)
    rng, construct_rng = jax.random.split(rng)
    flow_embed_model = pretrain_agents[config.agent].create(
        rng=construct_rng,
        observations=target_batch,
        encoder=encoder_def,
        decoder=decoder_def,
        **config.agent_kwargs,
    )
    flow_embed_model = checkpoints.restore_checkpoint(
        '/home/shivin/foundation_models/experiments/baselines_oxe/flow_vae_rn18_oxe_magic_soup_subset_h8_20250102_140803_1', 
        # '/home/shivin/foundation_models/experiments/baselines_oxe/flow_vae_rn18_libero_h8_prechunk_20250120_103634_1',
        target=flow_embed_model)

def load_br_embed_model(target_batch):
    # load br_embed model
    global br_embed_model
    config = get_config("br_vae")

    # define encoder and decoder
    encoder_def = encoders[config.encoder](**config.encoder_kwargs)
    decoder_def = decoders[config.decoder](**config.decoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(config.seed)
    rng, construct_rng = jax.random.split(rng)
    br_embed_model = pretrain_agents[config.agent].create(
        rng=construct_rng,
        observations=target_batch,
        encoder=encoder_def,
        decoder=decoder_def,
        **config.agent_kwargs,
    )
    br_embed_model = checkpoints.restore_checkpoint(
        '/home/shivin/foundation_models/experiments/baselines_oxe/br_vae_rn18_oxe_magic_soup_subset_h8_20250101_233131_1',
        # '/home/shivin/foundation_models/experiments/baselines_oxe/br_vae_rn18_libero_h8_prechunk_20250120_190454_1',
        target=br_embed_model)

def compute_embeddings(data_iter, outpath):
    tf.io.gfile.makedirs(os.path.dirname(outpath))
    with tf.io.TFRecordWriter(outpath) as writer:
        pbar = tqdm.tqdm(total=None)
        while True:
            try:
                batch = next(data_iter)
                # print_nested_dict(batch)

                flow_embeddings = compute_flow_embeddings(batch)
                br_embeddings = compute_br_embeddings(batch)
                batch['flow_embedding'] = np.asarray(flow_embeddings)
                batch['br_embedding'] = np.asarray(br_embeddings)

                dino_embeddings = compute_dino_embeddings(batch)
                batch['dino_embeddings'] = np.asarray(dino_embeddings)

                batch['observation']['image_primary'] = batch['image_primary_encoding']
                batch['observation']['image_wrist'] = batch['image_wrist_encoding']
                del batch['image_primary_encoding']
                del batch['image_wrist_encoding']
                del batch['future_image']

                for batch_idx in range(batch['action'].shape[0]):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=convert_batch_to_feature(batch, batch_idx)
                        )
                    )
                    writer.write(example.SerializeToString())

            except StopIteration:
                break
            pbar.update(1)

def main(_):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # load datasets
    tf.random.set_seed(0)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    train_data = EmbedDataset(
        data_name=FLAGS.data_name,
        data_dir=FLAGS.data_dir,
        dataset_statistics_path=FLAGS.data_stats_path,
        batch_size=FLAGS.batch_size,
        traj_transform_threads=24,
        traj_read_threads=24,
        frame_transforms_threads=8,
        act_pred_horizon=FLAGS.act_pred_horizon,
        future_image_horizon=FLAGS.future_image_horizon,
        window_size=WINDOW_SIZE,
        # load_split='train', 
        load_split='all',
    )
    train_iter = train_data.iterator()

    # load models
    load_gm_flow_model()
    outpath = os.path.join(FLAGS.out_dir, f"{FLAGS.data_name}_chunk{FLAGS.act_pred_horizon}_prechunk", "out.tfrecord")
    compute_embeddings(train_iter, outpath)

if __name__ == "__main__":
    app.run(main)

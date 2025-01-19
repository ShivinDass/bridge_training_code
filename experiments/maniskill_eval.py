from collections import defaultdict
import json
import os
import signal
import time
import numpy as np
from typing import Annotated, Optional

import torch
import tree
from mani_skill.utils import common
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *
from mani_skill.envs.sapien_env import BaseEnv
from pathlib import Path

import jax
from jaxrl_m.agents import agents
from jaxrl_m.vision import encoders
from flax.training import checkpoints
from torch.utils import dlpack as torch_dlpack
from jax import dlpack as jax_dlpack
import jax.numpy as jnp
from functools import partial

from absl import app, flags, logging
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string("ckpt_path", "", "path to ckpt file")
flags.DEFINE_string("env_id", "", "Name of the task to evaluate on. ex. PutCarrotOnPlateInScene-v1")
flags.DEFINE_integer("num_envs", 1, "Number of environments to use")
flags.DEFINE_integer("num_episodes", 100, "Number of episodes to run and record evaluation metrics over")
flags.DEFINE_integer("seed", 0, "Seed the model and environment. Default seed is 0")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)

from simpler_env.utils.action.action_ensemble import ActionEnsembler
from mani_skill.utils.geometry import rotation_conversions
class InferenceWrapper:

    def __init__(self, policy_fn):
        self.policy_fn = policy_fn

        self.action_ensembler = ActionEnsembler(4, 0)

    def reset(self):
        self.action_ensembler.reset()

    def get_action(self, obs):
        raw_actions = self.policy_fn(obs)
        raw_actions = self.action_ensembler.ensemble_action(raw_actions)

        raw_actions = jax2torch(raw_actions)
        raw_action = {
            "world_vector": raw_actions[:, :3],
            "rotation_delta": raw_actions[:, 3:6],
            "open_gripper": raw_actions[:, 6:7],  # range [0, 1]; 1 = open; 0 = close
        }
        raw_action = common.to_tensor(raw_action)

        action = {}
        action["world_vector"] = raw_action["world_vector"]
        action["rot_axangle"] = rotation_conversions.matrix_to_axis_angle(rotation_conversions.euler_angles_to_matrix(raw_action["rotation_delta"], "XYZ"))
        action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )
        
        return torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1)

def torch2jax(x_torch):
    x_torch = x_torch.contiguous() # https://github.com/google/jax/issues/8082
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax

def jax2torch(x_jax):
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch

def _resize_image(image: np.ndarray) -> np.ndarray:
    """resize image to a square image of size self.image_size. image should be shape (B, H, W, 3)"""
    image = jax.vmap(partial(jax.image.resize, shape=(128, 128, 3), method="lanczos3", antialias=True))(image)
    image = jnp.clip(jnp.round(image), 0, 255).astype(jnp.uint8)
    return image

def load_checkpoint():
    # create encoder from wandb config
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    act_pred_horizon = FLAGS.config.dataset_kwargs.get("act_pred_horizon")
    obs_horizon = FLAGS.config.dataset_kwargs.get("obs_horizon")

    # Set action
    example_actions = np.zeros((1, act_pred_horizon, 7), dtype=np.float32)

    # Set observations
    img_obs_shape = (1, obs_horizon, 128, 128, 3)
    example_obs = {"image": np.zeros(img_obs_shape, dtype=np.uint8)}

    # create agent from wandb config
    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_obs,
        actions=example_actions,
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )

    # load action metadata from wandb
    action_proprio_metadata = FLAGS.bridgedata_config["action_proprio_metadata"]
    action_mean = np.array(action_proprio_metadata["action"]["mean"])
    action_std = np.array(action_proprio_metadata["action"]["std"])

    # hydrate agent with parameters from checkpoint
    # agent = checkpoints.restore_checkpoint(FLAGS.ckpt_path, agent)
    agent = checkpoints.restore_checkpoint(FLAGS.ckpt_path, agent)#, step=int(8e4))

    def get_action(image):
        image = torch2jax(image)
        image = _resize_image(image)[:, None]
        # print(image.shape, image.dtype, image.min(), image.max())
        obs = {'image': image}
        nonlocal rng
        rng, key = jax.random.split(rng)
        action = jax.device_get(
            agent.sample_actions(obs, seed=key)
        )
        action = action * action_std + action_mean
        return action

    return get_action

def main(_):
    policy_fn = load_checkpoint()
    inference_model = InferenceWrapper(policy_fn)

    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)


    sensor_configs = dict()
    sensor_configs["shader_pack"] = 'default'
    env: BaseEnv = gym.make(
        FLAGS.env_id,
        obs_mode="rgb+segmentation",
        num_envs=FLAGS.num_envs,
        sensor_configs=sensor_configs
    )
    sim_backend = 'gpu' if env.device.type == 'cuda' else 'cpu'
    
    # from simpler_env.policies.rt1.rt1_model import RT1Inference
    # from simpler_env.policies.octo.octo_model import OctoInference
    # if len(args.ckpt_path) > 0 and "octo" in args.model:
    #     from octo.model.octo_model import OctoModel
    #     print(f"==> Loading Octo model from {args.ckpt_path}")
    #     octo_model = OctoModel.load_pretrained(args.ckpt_path, step=args.step)
    #     model = OctoInference(model=octo_model, model_type=args.model, policy_setup=policy_setup, init_rng=args.seed, action_scale=1)
    # elif args.model is not None:
    #     raise ValueError(f"Model {args.model} does not exist / is not supported.")


    exp_dir = os.path.join("videos", f"real2sim_eval/{FLAGS.config.agent}_{FLAGS.env_id}_seed{FLAGS.seed}")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    eval_metrics = defaultdict(list)
    eps_count = 0

    print(f"Running Real2Sim Evaluation of model {FLAGS.config.agent} on environment {FLAGS.env_id}")
    print(f"Using {FLAGS.num_envs} environments on the {sim_backend} simulation backend")
    
    while eps_count < FLAGS.num_episodes:
        seed = FLAGS.seed + eps_count
        obs, _ = env.reset(seed=seed, options={"episode_id": torch.tensor([seed + i for i in range(FLAGS.num_envs)])})
        # instruction = env.unwrapped.get_language_instruction()
        # print("instruction:", instruction[0])
        inference_model.reset()

        images = []
        predicted_terminated, truncated = False, False
        images.append(get_image_from_maniskill3_obs_dict(env, obs))
        
        while not (predicted_terminated or truncated):
            action = inference_model.get_action(images[-1])

            obs, reward, terminated, truncated, info = env.step(action)
            info = common.to_numpy(info)
            
            truncated = bool(truncated.any()) # note that all envs truncate and terminate at the same time.
            images.append(get_image_from_maniskill3_obs_dict(env, obs))

        for k, v in info.items():
            eval_metrics[k].append(v.flatten())
        for i in range(1):#range(len(images[-1])):
            images_to_video([img[i].cpu().numpy() for img in images], exp_dir, f"{sim_backend}_eval_{seed + i}_success={info['success'][i].item()}", fps=10, verbose=False)
        eps_count += FLAGS.num_envs
        if FLAGS.num_envs == 1:
            print(f"Evaluated episode {eps_count}. Seed {seed}. Results after {eps_count} episodes:")
        else:
            print(f"Evaluated {FLAGS.num_envs} episodes, seeds {seed} to {eps_count}. Results after {eps_count} episodes:")
        for k, v in eval_metrics.items():
            print(f"{k}: {np.mean(v)}")
    # Print timing information
    mean_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
    mean_metrics["total_episodes"] = eps_count
    
    metrics_path = os.path.join(exp_dir, f"{sim_backend}_eval_metrics.json")
    with open(metrics_path, "w") as f:
        mean_metrics['seed'] = FLAGS.seed
        json.dump(mean_metrics, f, indent=4)
    print(f"Evaluation complete. Results saved to {exp_dir}. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    app.run(main)
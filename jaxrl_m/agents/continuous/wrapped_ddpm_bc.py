import copy
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax

from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainStateWithBatchStats, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper

from jaxrl_m.networks.diffusion_nets import (
    FourierFeatures,
    cosine_beta_schedule,
    vp_beta_schedule,
    WrappedScoreActor,
)
from jaxrl_m.networks.mlp import MLP, MLPResNet
from jaxrl_m.vision.resnet_dec import resnetdec_configs


def ddpm_bc_loss(noise_prediction, noise, mask):
    ddpm_loss = jnp.where(mask[:, None], jnp.square(noise_prediction - noise).sum(-1), 0)
    valid_sample = mask.sum()
    if len(noise_prediction.shape) == 3:
        act_pred_horizon = noise_prediction.shape[1]
    else:
        act_pred_horizon = 1

    return (
        ddpm_loss.sum() / (valid_sample * act_pred_horizon),
        {"ddpm_loss": ddpm_loss, "ddpm_loss_mean": ddpm_loss.sum() / (valid_sample * act_pred_horizon)},
    )


class WrappedDDPMBCAgent(flax.struct.PyTreeNode):
    """
    Models action distribution with a diffusion model.

    Assumes observation histories as input and action sequences as output.
    """

    state: JaxRLTrainStateWithBatchStats
    config: dict = nonpytree_field()
    lr_schedule: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, batch_stats, rng):
            key, rng = jax.random.split(rng)
            time = jax.random.randint(
                key, (batch["actions"].shape[0],), 0, self.config["diffusion_steps"]
            )
            key, rng = jax.random.split(rng)
            noise_sample = jax.random.normal(key, batch["actions"].shape)

            alpha_hats = self.config["alpha_hats"][time]
            time = time[:, None]
            alpha_1 = jnp.sqrt(alpha_hats)[:, None, None]
            alpha_2 = jnp.sqrt(1 - alpha_hats)[:, None, None]

            noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

            rng, key = jax.random.split(rng)
            (embs, noise_pred) = self.state.apply_fn(
                {"params": params, "batch_stats": batch_stats},  # gradient flows through here
                batch["observations"],
                noisy_actions,
                time,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )
            bc_loss, info = ddpm_bc_loss(noise_pred, noise_sample, batch["action_loss_mask"])

            # decoder_outputs = self.state.apply_fn(
            #     {"params": params},
            #     embs,
            #     name="decoder",
            # )
            # recon_loss = ((decoder_outputs - batch["image_flows"]) ** 2).mean()
            # info["recon_loss"] = recon_loss

            return (
                bc_loss,# + self.config["recon_loss_lambda"] * recon_loss,
                info,
            )

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: PRNGKey = None,
        temperature: float = 1.0,
        argmax: bool = False,
        clip_sampler: bool = True,
    ) -> jnp.ndarray:
        assert len(observations["image"].shape) > 3, "Must use observation histories"

        def fn(input_tuple, time):
            current_x, rng = input_tuple
            input_time = jnp.broadcast_to(time, (current_x.shape[0], 1))

            _, eps_pred = self.state.apply_fn(
                {"params": self.state.target_params, "batch_stats": self.state.batch_stats},
                observations,
                current_x,
                input_time,
                name="actor",
            )

            alpha_1 = 1 / jnp.sqrt(self.config["alphas"][time])
            alpha_2 = (1 - self.config["alphas"][time]) / (
                jnp.sqrt(1 - self.config["alpha_hats"][time])
            )
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(key, shape=current_x.shape)
            z_scaled = temperature * z
            current_x = current_x + (time > 0) * (
                jnp.sqrt(self.config["betas"][time]) * z_scaled
            )

            if clip_sampler:
                current_x = jnp.clip(
                    current_x, self.config["action_min"], self.config["action_max"]
                )

            return (current_x, rng), ()

        key, rng = jax.random.split(seed)

        if len(observations["image"].shape) == 4:
            # unbatched input from evaluation
            batch_size = 1
            observations = jax.tree_map(lambda x: x[None], observations)
        else:
            batch_size = observations["image"].shape[0]

        input_tuple, () = jax.lax.scan(
            fn,
            (jax.random.normal(key, (batch_size, *self.config["action_dim"])), rng),
            jnp.arange(self.config["diffusion_steps"] - 1, -1, -1),
        )

        # NOTE: Don't know what is this for, but we won't be using this anyways
        for _ in range(self.config["repeat_last_step"]):
            input_tuple, () = fn(input_tuple, 0)

        action_0, rng = input_tuple

        if batch_size == 1:
            # this is an evaluation call so unbatch
            return action_0[0]
        else:
            return action_0

    @jax.jit
    def get_predicted_flow(self, observations):
        x = jax.random.normal(jax.random.PRNGKey(0), (1, *self.config["action_dim"]))
        time = jnp.broadcast_to(0, (x.shape[0], 1))

        embs, _ = self.state.apply_fn(
            {"params": self.state.params, "batch_stats": self.state.batch_stats},
            observations,
            x,
            time,
            name="actor",
        )

        decoder_outputs = self.state.apply_fn(
            {"params": self.state.params},
            embs,
            name="decoder",
        )
        return decoder_outputs

    @jax.jit
    def get_debug_metrics(self, batch, seed, gripper_close_val=None):
        actions = self.sample_actions(
            observations=batch["observations"], seed=seed
        )

        metrics = {"mse": ((actions - batch["actions"]) ** 2).sum((-2, -1)).mean()}

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        use_proprio: bool = False,
        score_network_kwargs: dict = {
            "time_dim": 32,
            "num_blocks": 3,
            "dropout_rate": 0.1,
            "hidden_dim": 256,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        decay_steps: Optional[int] = None,
        # Algorithm config
        beta_schedule: str = "cosine",
        diffusion_steps: int = 25,
        action_samples: int = 1,
        repeat_last_step: int = 0,
        target_update_rate=0.002,
        dropout_target_networks=True,
        recon_loss_lambda=0.01
    ):
        assert len(actions.shape) > 1, "Must use action chunking"
        assert len(observations["image"].shape) > 3, "Must use observation histories"

        encoder_def = EncodingWrapper(
            encoder=encoder_def, use_proprio=use_proprio, stop_gradient=False
        )

        decoder_def = resnetdec_configs["resnet-18-dec"](num_output_channels=2, output_hw=128) #

        networks = {
            "actor": WrappedScoreActor(
                encoder_def,
                FourierFeatures(score_network_kwargs["time_dim"], learnable=True),
                MLP(
                    (
                        2 * score_network_kwargs["time_dim"],
                        score_network_kwargs["time_dim"],
                    )
                ),
                MLPResNet(
                    score_network_kwargs["num_blocks"],
                    actions.shape[-2] * actions.shape[-1],
                    dropout_rate=score_network_kwargs["dropout_rate"],
                    use_layer_norm=score_network_kwargs["use_layer_norm"],
                ),
            ),
            "decoder": decoder_def #
        }

        model_def = ModuleDict(networks)

        rng, init_rng = jax.random.split(rng)
        if len(actions.shape) == 3:
            example_time = jnp.zeros((actions.shape[0], 1))
        else:
            example_time = jnp.zeros((1,))
        params_and_batch_stats = model_def.init(
            init_rng, actor=[observations, actions, example_time], decoder=[jnp.ones((1, 512))] #
        )
        params = params_and_batch_stats["params"]
        batch_stats = params_and_batch_stats["batch_stats"]

        # no decay
        # lr_schedule = optax.warmup_cosine_decay_schedule(
        #     init_value=0.0,
        #     peak_value=learning_rate,
        #     warmup_steps=warmup_steps,
        #     decay_steps=warmup_steps + 1,
        #     end_value=learning_rate,
        # )
        # lr_schedules = {"actor": lr_schedule, "decoder": lr_schedule}
        # if actor_decay_steps is not None:
        #     lr_schedules["actor"] = optax.warmup_cosine_decay_schedule(
        #         init_value=0.0,
        #         peak_value=learning_rate,
        #         warmup_steps=warmup_steps,
        #         decay_steps=actor_decay_steps,
        #         end_value=0.0,
        #     )
        # txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        if decay_steps is None:
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=warmup_steps + 1,
                end_value=learning_rate,
            )
        else:
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=0.0,
            )
        tx = optax.adam(lr_schedule)

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainStateWithBatchStats.create(
            apply_fn=model_def.apply,
            params=params,
            # txs=txs,
            txs=tx,
            target_params=params,
            batch_stats=batch_stats,
            rng=create_rng,
        )

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(diffusion_steps))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, diffusion_steps)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(diffusion_steps))

        alphas = 1 - betas
        alpha_hat = jnp.array(
            [jnp.prod(alphas[: i + 1]) for i in range(diffusion_steps)]
        )

        config = flax.core.FrozenDict(
            dict(
                target_update_rate=target_update_rate,
                dropout_target_networks=dropout_target_networks,
                action_dim=actions.shape[-2:],
                action_max=2.0,
                action_min=-2.0,
                betas=betas,
                alphas=alphas,
                alpha_hats=alpha_hat,
                diffusion_steps=diffusion_steps,
                action_samples=action_samples,
                repeat_last_step=repeat_last_step,
                recon_loss_lambda=recon_loss_lambda
            )
        )
        return cls(state, config, lr_schedule)

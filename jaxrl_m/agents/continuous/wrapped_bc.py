from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
from jaxrl_m.common.encoding import EncodingWrapper
import numpy as np
import flax
import flax.linen as nn
import optax

from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainStateWithBatchStats, ModuleDict, nonpytree_field
from jaxrl_m.networks.actor_critic_nets import WrappedPolicy
from jaxrl_m.networks.mlp import MLP
from jaxrl_m.vision.resnet_dec import resnetdec_configs


class WrappedBCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainStateWithBatchStats
    lr_schedule: Any = nonpytree_field()
    recon_loss_lambda: float

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, batch_stats, rng):
            rng, key = jax.random.split(rng)
            (embs, dist), variables = self.state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                batch["observations"],
                mutable=["batch_stats"],
                temperature=1.0,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            decoder_outputs = self.state.apply_fn(
                {"params": params},
                embs,
                name="decoder",
            )
            recon_loss = ((decoder_outputs - batch["image_flows"]) ** 2).mean()

            return (
                actor_loss + self.recon_loss_lambda * recon_loss,
                {
                    "actor_loss": actor_loss,
                    "mse": mse.mean(),
                    "log_probs": log_probs.mean(),
                    "pi_actions": pi_actions.mean(),
                    "mean_std": actor_std.mean(),
                    "max_std": actor_std.max(),
                    "recon_loss": recon_loss,
                    "batch_stats": variables['batch_stats'],
                },
            )

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        argmax=False
    ) -> jnp.ndarray:
        _, dist = self.state.apply_fn(
            {"params": self.state.params, "batch_stats": self.state.batch_stats},
            observations,
            temperature=temperature,
            name="actor",
        )
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_predicted_flow(self, observations):
        embs, _ = self.state.apply_fn(
            {"params": self.state.params, "batch_stats": self.state.batch_stats},
            observations,
            temperature=1.0,
            name="actor",
        )

        decoder_outputs = self.state.apply_fn(
            {"params": self.state.params},
            embs,
            name="decoder",
        )
        return decoder_outputs

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        embs, dist = self.state.apply_fn(
            {"params": self.state.params, "batch_stats": self.state.batch_stats},
            batch["observations"],
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
        decoder_outputs = self.state.apply_fn(
            {"params": self.state.params},
            embs,
            name="decoder",
        )
        recon_loss = ((decoder_outputs - batch["image_flows"]) ** 2).mean()        

        return {"mse": mse, "log_probs": log_probs, "pi_actions": pi_actions, "recon_loss": recon_loss}

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        use_proprio: bool = False,
        network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "state_dependent_std": False,
            "dropout": 0.0,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
        # Aux loss
        recon_loss_lambda: float = 0.01,
    ):
        encoder_def = EncodingWrapper(
            encoder=encoder_def, use_proprio=use_proprio, stop_gradient=False
        )

        decoder_def = resnetdec_configs["resnet-18-dec"](num_output_channels=2, output_hw=128) 

        network_kwargs["activate_final"] = True
        networks = {
            "actor": WrappedPolicy(
                encoder_def,
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs
            ),
            "decoder": decoder_def,
        }

        model_def = ModuleDict(networks)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        # NOTE: use a fixed emb size 512 for now
        params_and_batch_stats = model_def.init(init_rng, actor=[observations], decoder=[jnp.ones((1, 512 * 2))])
        params = params_and_batch_stats["params"]
        batch_stats = params_and_batch_stats["batch_stats"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainStateWithBatchStats.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            batch_stats=batch_stats,
            rng=create_rng,
        )

        return cls(state, lr_schedule, recon_loss_lambda)

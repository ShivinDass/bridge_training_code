from functools import partial
import distrax
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from absl import logging
# from clu import parameter_overview

from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict
from jaxrl_m.networks.vae import VAEEncoder, VAEDecoder
from jaxrl_m.networks.mlp import MLP


class OpticalFlowVAEAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    log_std_min: float = -6
    log_std_max: float = 6
    kl_weight: float = 1e-4

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            posterior_params = self.state.apply_fn(
                {"params": params},
                batch["image_flows"],
                name="encoder",
            )

            means, log_stds = jnp.split(posterior_params, 2, axis=-1)
            log_stds=jnp.clip(log_stds, self.log_std_min, self.log_std_max)
            dist = distrax.MultivariateNormalDiag(
                loc=means, scale_diag=jnp.exp(log_stds)
            )

            samples = dist.sample(seed=key)

            reconstructions = self.state.apply_fn(
                {"params": params},
                samples,
                name="decoder",
            )

            recon_loss = ((reconstructions - batch["image_flows"]) ** 2).mean()
            # closed-form Gaussian KL from N(0, 1) prior
            kl_loss = (-0.5 * (1. + log_stds - means ** 2 - jnp.exp(log_stds)).sum(-1)).mean()

            vae_loss = recon_loss + self.kl_weight * kl_loss
            return (
                vae_loss,
                {
                    "vae_loss": vae_loss,
                    "recon_loss": recon_loss,
                    "kl_loss": kl_loss,
                    # "samples": samples,
                }
            )
            
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        return self.replace(state=new_state), info

    @jax.jit
    def get_debug_metrics(self, batch: Batch, seed: PRNGKey):
        posterior_params = self.state.apply_fn(
            {"params": self.state.params},
            batch["image_flows"],
            name="encoder",
        )

        means, log_stds = jnp.split(posterior_params, 2, axis=-1)
        log_stds=jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

        samples = dist.sample(seed=seed)

        reconstructions = self.state.apply_fn(
            {"params": self.state.params},
            samples,
            name="decoder",
        )

        recon_loss = ((reconstructions - batch["image_flows"]) ** 2).mean()
        # closed-form Gaussian KL from N(0, 1) prior
        kl_loss = (-0.5 * (1. + log_stds - means ** 2 - jnp.exp(log_stds)).sum(-1)).mean()

        vae_loss = recon_loss + self.kl_weight * kl_loss
        return {"vae_loss": vae_loss, "recon_loss": recon_loss, "kl_loss": kl_loss} 

    @jax.jit
    def visualize_reconstruction(self, batch: Batch, seed: PRNGKey):
        posterior_params = self.state.apply_fn(
            {"params": self.state.params},
            batch["image_flows"],
            name="encoder",
        )

        means, log_stds = jnp.split(posterior_params, 2, axis=-1)
        log_stds=jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

        samples = dist.sample(seed=seed)

        reconstructions = self.state.apply_fn(
            {"params": self.state.params},
            samples,
            name="decoder",
        )

        return reconstructions

    @jax.jit
    def compute_embeddings(self, batch: Batch):
        posterior_params = self.state.apply_fn(
            {"params": self.state.params},
            batch["image_flows"],
            name="encoder",
        )
        means, _ = jnp.split(posterior_params, 2, axis=-1)
        return means

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        # Model architecture
        encoder: nn.Module,
        decoder: nn.Module,
        latent_kwargs: dict = {
            "hidden_dims": [128, 300, 400],
            "output_dim": 128
        },
        vae_kwargs: dict = {
            "log_std_max": 6,
            "log_std_min": -6,
            "kl_weight": 1e-4,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
    ):
        mlp_encoder_kwargs = {
            "hidden_dims": latent_kwargs["hidden_dims"] + [latent_kwargs["output_dim"] * 2],
            "activations": nn.relu,
        }
        latent_encoder = MLP(**mlp_encoder_kwargs)
        mlp_decoder_kwargs = {
            "hidden_dims": latent_kwargs["hidden_dims"],
            "activations": nn.relu,
            "activate_final": True,
        }
        latent_decoder = MLP(**mlp_decoder_kwargs)

        networks = {
            "encoder": VAEEncoder(
                encoder,
                latent_encoder,
            ),
            "decoder": VAEDecoder(
                latent_decoder,
                decoder,
            )
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
        # params = model_def.init(init_rng, encoder=[observations["image_flows"]], decoder=[jnp.zeros((1, latent_kwargs["output_dim"]))])["params"]
        variables = model_def.init(init_rng, encoder=[observations["image_flows"]], decoder=[jnp.zeros((1, latent_kwargs["output_dim"]))])
        # logging.info(parameter_overview.get_parameter_overview(variables))
        params = variables["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        return cls(state, **vae_kwargs)


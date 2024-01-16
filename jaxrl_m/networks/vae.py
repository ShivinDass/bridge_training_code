import jax.numpy as jnp
import flax.linen as nn
import einops


class VAEEncoder(nn.Module):
    encoder: nn.Module
    latent_encoder: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray):
        x = self.encoder(observations)
        x = einops.rearrange(x, "... h w c -> ... (h w c)")
        x = self.latent_encoder(x)
        return x


class VAEDecoder(nn.Module):
    latent_decoder: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(self, embs: jnp.ndarray):
        x = self.latent_decoder(embs)
        x = self.decoder(x)
        return x
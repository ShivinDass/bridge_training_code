import jax.numpy as jnp
import flax.linen as nn
import einops


class VAEEncoder(nn.Module):
    encoder: nn.Module
    latent_encoder: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray):
        x = self.encoder(observations)
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
    
class VAEImageActionEncoder(nn.Module):
    encoder: nn.Module
    latent_encoder: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray):
        '''
            observations: (B, 128)
            actions: (B, 8, 7)
        '''
        
        x = self.encoder(observations)
        
        actions = actions[:, :4].reshape(actions.shape[0], -1) # actions: (B, 8, 7) -> (B, 4, 7) -> (B, 28)
        x = jnp.concatenate([x, actions], axis=-1)

        x = self.latent_encoder(x)
        return x
    

class VAEImageActionDecoder(nn.Module):
    latent_decoder: nn.Module
    decoder: nn.Module
    action_decoder: nn.Module

    @nn.compact
    def __call__(self, embs: jnp.ndarray):
        x = self.latent_decoder(embs)
        recon_action = self.action_decoder(x)
        recon_image = self.decoder(x)
        return recon_image, recon_action
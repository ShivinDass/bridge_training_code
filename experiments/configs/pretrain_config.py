from ml_collections import ConfigDict

num_steps = int(5e5)
def get_config(config_string):
    base_real_config = dict(
        batch_size=128,
        num_steps=num_steps,
        log_interval=500,
        eval_interval=num_steps//10,
        save_interval=num_steps//10,
        save_dir="/home/shivin/foundation_models/experiments/",
        data_path="",
        resume_path=None,
        seed=42,
    )

    base_data_config = dict(
        shuffle_buffer_size=25000,
        augment=False,
    )

    possible_structures = {
        "optical_flow_vae": ConfigDict(
            dict(
                agent="optical_flow_vae",
                agent_kwargs=dict(
                    latent_kwargs=dict(
                        hidden_dims=[300, 400],
                        output_dim=128
                    ),
                    vae_kwargs=dict(
                        log_std_min=-4,
                        log_std_max=15,
                        kl_weight=1e-4,
                    ),
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=num_steps,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                encoder="resnetv1-18",
                encoder_kwargs=dict(
                    pooling_method="proj",
                    projection_size=128,
                    normalized=False,
                ),
                decoder="resnet-18-dec",
                decoder_kwargs=dict(
                    num_output_channels=2, output_hw=128
                ),
                **base_real_config,
            )
        ),

        "br_vae": ConfigDict(
            dict(
                agent="br_vae",
                agent_kwargs=dict(
                    latent_kwargs=dict(
                        hidden_dims=[300, 400],
                        output_dim=128
                    ),
                    vae_kwargs=dict(
                        log_std_min=-4,
                        log_std_max=15,
                        kl_weight=1e-4,
                    ),
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=num_steps,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                encoder="resnetv1-18",
                encoder_kwargs=dict(
                    pooling_method="proj",
                    projection_size=128,
                    normalized=False, # True
                ),
                decoder="resnet-18-dec",
                decoder_kwargs=dict(
                    num_output_channels=3, output_hw=128
                ),
                **base_real_config,
            )
        ),
    }

    return possible_structures[config_string]

from ml_collections import ConfigDict


def get_config(config_string):
    base_real_config = dict(
        batch_size=256,
        num_steps=int(1e5),
        log_interval=500,
        eval_interval=5000,
        save_interval=25000,
        save_dir="/iliad/u/lhlin/bridge_data_v2/experiment_logs",
        data_path="/iliad/u/lhlin/bridge_data_v2/datasets_tfrecord_flow",
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
                        projection_size=128,
                        hidden_dims=[300, 400],
                        output_dim=128
                    ),
                    vae_kwargs=dict(
                        log_std_min=-4,
                        log_std_max=15,
                        kl_weight=1e-4,
                    ),
                    learning_rate=1e-3,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                encoder="resnetv1-18",
                encoder_kwargs=dict(
                    pooling_method="none"
                ),
                decoder="resnet-18-dec",
                decoder_kwargs=dict(
                    num_output_channels=2, output_hw=128
                ),
                **base_real_config,
            )
        ),
    }

    return possible_structures[config_string]

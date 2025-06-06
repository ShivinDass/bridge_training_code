from ml_collections import ConfigDict


def get_config(config_string):
    base_real_config = dict(
        batch_size=256,
        # num_steps=int(2e6),
        num_steps=300000,
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir="/iliad/u/lhlin/bridge_data_v2/experiment_logs",
        data_path="/iliad/u/lhlin/bridge_data_v2/datasets_tfrecord_flow",
        # data_path="/tmp/bridgedata_v2/tf_flow", # for iliad 5, 6
        resume_path=None,
        seed=42,
    )

    base_data_config = dict(
        shuffle_buffer_size=25000,
        augment=True,
        augment_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
    )

    possible_structures = {
        "gc_ddpm_bc": ConfigDict(
            dict(
                agent="gc_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    obs_horizon=1,
                    act_pred_horizon=1,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=True, act="swish"
                ),
                **base_real_config,
            )
        ),
        "flow_ddpm_bc_pretrained-freezed-BN_na8_prechunk": ConfigDict(
            dict(
                agent="flow_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                    recon_loss_lambda=0.01,
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=False, # Don't use the observations to relabel the actions since we already prechunk the actions
                    obs_horizon=1,
                    act_pred_horizon=8,
                    prechunk_act=True,
                    **base_data_config,
                ),
                encoder="pretrained_resnet34",
                encoder_kwargs=dict(freezed_BN=True),
                **base_real_config,
            )
        ),
        "gc_ddpm_bc_pretrained-freezed-BN_na8": ConfigDict(
            dict(
                agent="gc_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    early_goal_concat=False,
                    shared_goal_encoder=False,
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    obs_horizon=1,
                    act_pred_horizon=8,
                    **base_data_config,
                ),
                encoder="pretrained_resnet34",
                encoder_kwargs=dict(freezed_BN=True),
                **base_real_config,
            )
        ),
        "ddpm_bc_pretrained-freezed-BN_na8": ConfigDict(
            dict(
                agent="flow_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                    recon_loss_lambda=0,
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    obs_horizon=1,
                    act_pred_horizon=8,
                    **base_data_config,
                ),
                encoder="pretrained_resnet34",
                encoder_kwargs=dict(freezed_BN=True),
                **base_real_config,
            )
        ),
        "ddpm_bc_pretrained-freezed-BN_na8_prechunk": ConfigDict(
            dict(
                agent="flow_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                    recon_loss_lambda=0,
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=False,
                    obs_horizon=1,
                    act_pred_horizon=8,
                    prechunk_act=True,
                    **base_data_config,
                ),
                encoder="pretrained_resnet34",
                encoder_kwargs=dict(freezed_BN=True),
                **base_real_config,
            )
        ),
    }

    return possible_structures[config_string]

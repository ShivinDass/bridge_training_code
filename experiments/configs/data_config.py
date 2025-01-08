from absl import logging
import ml_collections

ACT_MEAN = [
    ###### Bridge default #####
    # 1.9296819e-04,
    # 1.3667766e-04,
    # -1.4583133e-04,
    # -1.8390431e-04,
    # -3.0808983e-04,
    # 2.7425270e-04,
    # 5.9716219e-01,

    ##### Viper_x_pot #####
    0.00227883,
    0.00154444,
    -0.00229705,
    -0.0013669,
    -0.00198731,
    0.00193355,
    0.65677481
]

ACT_STD = [
    ###### Bridge default #####
    # 0.00912848,
    # 0.0127196,
    # 0.01229497,
    # 0.02606696,
    # 0.02875283,
    # 0.07807977,
    # 0.48710242,

    ##### Viper_x_pot #####
    0.00990546,
    0.01049463,
    0.01018103,
    0.00905978,
    0.01723201,
    0.01993144,
    0.47299017
]

# NOTE: we would not use proprioception anyways
PROPRIO_MEAN =[
    0.29414398,
    0.05308576,
    0.13111071,
    -0.066769,
    -0.12331675,
    0.11321178,
    0.72671092
]

PROPRIO_STD = [
    0.05813769,
    0.080115,
    0.06733975,
    0.06441016,
    0.10455791,
    0.14951433,
    0.34510486
]

ACTION_PROPRIO_METADATA = {
    "action": {
        "mean": ACT_MEAN,
        "std": ACT_STD,
        # TODO compute these
        # "min": ACT_MEAN,
        # "max": ACT_STD,
    },
    # TODO compute these
    "proprio": {
        "mean": PROPRIO_MEAN,
        "std": PROPRIO_STD,
        # "min": ACT_MEAN,
        # "max": ACT_STD
    },
}


def get_config(config_string):
    actual_dataset_map = {
        # "viper_x_pot": "viper_x_pot",
        "viper_x_pot_h8": "viper_x_pot_h8", # h8 means the optical flow is computed from t to t+8
        "viper_x_pot_h8_prechunk": "viper_x_pot_h8_prechunk",
        "viper_x_microwave_h8": "viper_x_microwave_h8",
        "viper_x_microwave_good_start_pos_h8": "viper_x_microwave_good-start-pos_h8", # good-start-pos means the data collection has the same initial position as evaluation
        "viper_x_microwave_good_start_pos_h8_prechunk": "viper_x_microwave_good-start-pos_h8_prechunk",
        "viper_x_microwave_good_start_pos_h8_prechunk_filtered": "viper_x_microwave_good-start-pos_h8_prechunk_filtered",
        "bridgedata_v2": "bridge_data_v2/?*/?*/?*",
        "bridgedata_v2_h8": "bridge_data_v2_h8/?*/?*/?*",
        # NOTE: exclude 0 because the processing is not finished
        "oxe_subset_h8": "flow_retrieval_subset_[1-7]_h8_prechunk",
        "oxe_broth": "oxe_flow_h4_prechunk",
        "oxe_magic_soup_subset_h8": "oxe_magic_soup_s?_h8_prechunk",
        # NOTE: just for testing
        "pot_microwave_vae_0.01": "pot_with-microwave-vae_bridge_data_v2_h8_0.01_prechunk",
    }
    general_dataset_map = {
        "flow_retrieved": "{}_bridge_data_v2_h8_{}_prechunk",
        "br_retrieved": "{}_bridge_data_v2_h8_br_{}_prechunk",
        "sailor_retrieved": "{}_bridge_data_v2_h8_sailor_{}_prechunk",
        "oxe_flow_retrieved": "{}_flow_retrieval_subset_\[1-7\]_h8_prechunk_{}_prechunk",
    }

    sample_weights = None
    if len(config_string.split('-')) > 1:
        sample_option = config_string.split('-')[1]
        if sample_option == "balance":
            sample_weights = [0.5, 0.5]
        else:
            raise ValueError(f"Unknown sample option: {sample_option}")

    dataset_config_string = config_string.split('-')[0]
    if len(dataset_config_string.split('+')) == 1:
        target_dataset = dataset_config_string
        include = [
            [actual_dataset_map[target_dataset]]
        ]
        included_in_action_loss = [True]
    else:
        target_dataset, prior_dataset = dataset_config_string.split('+')
        included_in_action_loss = [True, True]
        if actual_dataset_map.get(prior_dataset) is not None:
            include = [
                [actual_dataset_map[target_dataset]],
                [actual_dataset_map[prior_dataset]]
            ]
        else:
            # hacky
            if 'pot' in target_dataset:
                task = 'pot'
            elif 'microwave_good_start_pos' in target_dataset:
                task = 'microwave_good-start-pos'
            else:
                task = 'microwave'
                

            assert actual_dataset_map[target_dataset].endswith("prechunk")
            found = False
            for retrieval_method in general_dataset_map.keys():
                if prior_dataset.startswith(retrieval_method):
                    retrieval_threshold = prior_dataset.split('_')[-1]
                    include = [
                        [actual_dataset_map[target_dataset]],
                        [general_dataset_map[retrieval_method].format(task, retrieval_threshold)]
                    ]
                    found = True
                    break
            if not found:
                raise ValueError(f"Unknown dataset: {prior_dataset}")

    return ml_collections.ConfigDict(
        {
            "include": include,
            "exclude": [],
            "sample_weights": sample_weights,
            "action_proprio_metadata": ACTION_PROPRIO_METADATA,
            "dtype": "float16", #if target_dataset == "oxe_broth" else "float32",
            "included_in_action_loss": included_in_action_loss,
        }
    )
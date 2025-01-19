THRESHOLD=0.01

python experiments/retrieve.py \
--config experiments/configs/pretrain_config.py:optical_flow_vae \
--checkpoint_path /home/shivin/foundation_models/experiments/baselines_oxe/flow_vae_rn18_oxe_magic_soup_subset_h8_20250102_140803_1 \
--target_dataset_path simpler_carrot_flow_h8_prechunk \
--prior_dataset_path bridge_flow_h8_prechunk \
--prior_dataset_flow_dtype float16 \
--threshold $THRESHOLD \
--output_dir /mnt/hdd1/retrieved_simpler_carrot \
--prefix retrieved_simpler_carrot \
--act_pred_horizon 8 \
--prechunk True

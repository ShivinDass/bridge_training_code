THRESHOLD=0.01

python experiments/retrieve.py \
--config experiments/configs/pretrain_config.py:optical_flow_vae \
--checkpoint_path <...> \
--target_dataset_path <...> \
--prior_dataset_path <...> \
--prior_dataset_flow_dtype float16 \
--threshold $THRESHOLD \
--output_dir <...> \
--prefix <...> \
--act_pred_horizon 8 \
--prechunk True

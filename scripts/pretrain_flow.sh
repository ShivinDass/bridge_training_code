# DATA=oxe_magic_soup_subset_h8
# DATA_PATH="/mnt/hdd1/baselines/vae_training_data"
DATA=libero_h8_prechunk
DATA_PATH="/mnt/hdd2/libero90_optical_flow"

python experiments/pretrain_flow.py \
--config experiments/configs/pretrain_config.py:optical_flow_vae \
--data_config experiments/configs/data_config.py:${DATA} \
--name flow_vae_rn18_${DATA} \
--config.data_path ${DATA_PATH} \
--config.seed 1

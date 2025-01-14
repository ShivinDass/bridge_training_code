DATA=oxe_magic_soup_subset_h8
DATA_PATH="/mnt/hdd1/baselines/vae_training_data"

python experiments/pretrain_br.py \
--config experiments/configs/pretrain_config.py:br_vae \
--data_config experiments/configs/data_config.py:${DATA} \
--name br_vae_rn18_${DATA} \
--config.data_path ${DATA_PATH} \
--config.seed 1

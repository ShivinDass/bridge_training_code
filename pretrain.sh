DATA=...
DATA_PATH=...

python experiments/pretrain.py \
--config experiments/configs/pretrain_config.py:optical_flow_vae \
--bridgedata_config experiments/configs/data_config.py:${DATA} \
--name flow_vae_rn18_${DATA} \
--config.data_path ${DATA_PATH} \
--config.seed 1

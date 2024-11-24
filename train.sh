THRESHOLD=0.01

MODEL=flow_ddpm_bc_pretrained-freezed-BN_na8_prechunk
DATA=...

python experiments/train.py \
--config experiments/configs/train_config.py:${MODEL} \
--bridgedata_config experiments/configs/data_config.py:${DATA} \
--name <...> \
--config.seed 1

THRESHOLD=0.01

# MODEL=flow_ddpm_bc_pretrained-freezed-BN_na8_prechunk
MODEL=ddpm_bc_pretrained-freezed-BN_na8_prechunk
DATA=simpler_carrot+bridge_carrot_retrieved-balance
# DATA=bridge_carrot_retrieved+bridge_carrot_retrieved-balance

python experiments/train.py \
--config experiments/configs/train_config.py:${MODEL} \
--bridgedata_config experiments/configs/data_config.py:${DATA} \
--name bc_training_wo-recon \
--config.seed 1
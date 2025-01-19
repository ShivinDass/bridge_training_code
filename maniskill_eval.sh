MODEL=flow_ddpm_bc_pretrained-freezed-BN_na8_prechunk
# MODEL=ddpm_bc_pretrained-freezed-BN_na8_prechunk
DATA=simpler_carrot+bridge_carrot_retrieved-balance
# DATA=simpler_carrot+simpler_carrot-balance

# CKPT=/home/shivin/foundation_models/experiments/baselines_oxe/baselines_oxe/flow-retrieval_bc_training_20250114_025159_1
CKPT=/home/shivin/foundation_models/experiments/baselines_oxe/baselines_oxe/bc_training_wo-recon_20250115_015323_1

python experiments/maniskill_eval.py \
--config experiments/configs/train_config.py:${MODEL} \
--bridgedata_config experiments/configs/data_config.py:${DATA} \
--config.seed 1 \
--ckpt_path $CKPT \
--env_id PutCarrotOnPlateInScene-v1 \
--num_envs 5 \
--num_episodes 100 \
--seed 1
TASK=book-caddy #_discrete
# TASK=book-caddy_c0.1_bob60
# TASK=moka-moka

DATASET=libero90_horizon30
# DATASET=libero90

DM_PATH=/mnt/hdd2/dm_experiments/${DATASET}_128x128/${TASK}
PRIOR_DATASET_PATH=/mnt/hdd2/baselines/prior_data/${DATASET}_chunk8_prechunk

OUTPUT_DIR=/mnt/hdd2/baselines/retrieved_data_chunk8_dm/${DATASET}

python experiments/retrieve_dm.py \
--dm_path $DM_PATH \
--prior_dataset_path $PRIOR_DATASET_PATH \
--batch_size 256 \
--output_dir $OUTPUT_DIR \
--topk 0.1 \
--iter_dm 30 \
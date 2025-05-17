# TASK=book-caddy_weighted-3 #_discrete
# TASK=bowl-cabinet_weighted
TASK=book-caddy_weighted_subopt
# TASK=moka-moka_weighted
# TASK=cream-butter_weighted_subopt
# TASK=soup-sauce_weighted
# TASK=mug-microwave_fullval
# TASK=soup-cheese_weighted-3
# TASK=mug-mug_weighted
# TASK=stove-moka_weighted

# DATASET=libero90_horizon30
# DATASET=libero90
DATASET=libero90_horizon15


DM_PATH=/mnt/hdd2/dm_experiments/${DATASET}_128x128/${TASK}
PRIOR_DATASET_PATH=/mnt/hdd2/baselines/prior_data/${DATASET}_chunk8_prechunk

OUTPUT_DIR=/mnt/hdd2/baselines/retrieved_data_chunk8_dm/${DATASET}

python experiments/libero/retrieve_libero_dm.py \
--dm_path $DM_PATH \
--prior_dataset_path $PRIOR_DATASET_PATH \
--batch_size 256 \
--output_dir $OUTPUT_DIR \
--topk 0.05 \
--iter_dm 25 \
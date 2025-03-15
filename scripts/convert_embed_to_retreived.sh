TARGET_DATASET_PATH=/mnt/hdd1/baselines/target_data/easy_pick_dataset_n10_h8_prechunk
PRIOR_DATASET_PATH=/mnt/hdd1/baselines/embed_data_2

OUTPUT_DIR=/mnt/hdd1/baselines/retrieved_data/dm_ut_oxe_easy_pick_n10

python experiments/convert_embed_to_retrieved.py \
--target_dataset_path $TARGET_DATASET_PATH \
--prior_dataset_path $PRIOR_DATASET_PATH \
--batch_size 256 \
--output_dir $OUTPUT_DIR

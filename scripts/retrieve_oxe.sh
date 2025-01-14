TARGET_DATASET_PATH=/mnt/hdd1/baselines/target_data/simpler_carrot_dataset_h8_prechunk
PRIOR_DATASET_PATH=/mnt/hdd1/baselines/embed_data

OUTPUT_DIR=/mnt/hdd1/baselines/retrieved_data
THRESHOLD=0.9

python experiments/retrieve_oxe.py \
--target_dataset_path $TARGET_DATASET_PATH \
--prior_dataset_path $PRIOR_DATASET_PATH \
--threshold $THRESHOLD \
--batch_size 256 \
--output_dir $OUTPUT_DIR \
--prechunk True

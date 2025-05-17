# TARGET_DATASET_PATH=/mnt/hdd1/baselines/target_data/easy_pick_dataset_n10_chunk4_prechunk
# THRESHOLD=0.01

# TARGET_DATASET_PATH=/mnt/hdd1/baselines/target_data/deo_in_pouch_dataset_n20_chunk4_prechunk
# THRESHOLD=0.0075

# TARGET_DATASET_PATH=/mnt/hdd1/baselines/target_data/tiago_sink_dataset_n20_chunk4_prechunk
# THRESHOLD=0.005

TARGET_DATASET_PATH=/mnt/hdd1/baselines/target_data/droid_dataset_full_chunk4_prechunk
THRESHOLD=0.01

PRIOR_DATASET_PATH=/mnt/hdd1/baselines/embed_data
OUTPUT_DIR=/mnt/hdd1/baselines/retrieved_data

python experiments/oxe/retrieve_oxe_baselines.py \
--target_dataset_path $TARGET_DATASET_PATH \
--prior_dataset_path $PRIOR_DATASET_PATH \
--threshold $THRESHOLD \
--batch_size 256 \
--output_dir $OUTPUT_DIR \

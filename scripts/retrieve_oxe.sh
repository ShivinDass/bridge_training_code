TARGET_DATASET_PATH=/mnt/hdd1/baselines/target_data/easy_pick_dataset_n10_h8_prechunk
PRIOR_DATASET_PATH=/mnt/hdd1/baselines/embed_data_2

OUTPUT_DIR=/mnt/hdd1/baselines/retrieved_data
THRESHOLD=1.0

# for libero baselines
# TARGET_DATASET_PATH=/mnt/hdd2/baselines/target_data_chunk8/study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_h8_prechunk
# PRIOR_DATASET_PATH=/mnt/hdd2/baselines/prior_data/libero90_chunk8_prechunk

# OUTPUT_DIR=/mnt/hdd2/baselines/retrieved_data_chunk8
# THRESHOLD=0.1

python experiments/retrieve_oxe.py \
--target_dataset_path $TARGET_DATASET_PATH \
--prior_dataset_path $PRIOR_DATASET_PATH \
--threshold $THRESHOLD \
--batch_size 256 \
--output_dir $OUTPUT_DIR \

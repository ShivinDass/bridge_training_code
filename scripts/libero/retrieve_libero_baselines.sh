# for libero baselines
PRIOR_DATASET_PATH=/mnt/hdd2/baselines/prior_data/libero90_horizon15_chunk8_prechunk

OUTPUT_DIR=/mnt/hdd2/baselines/retrieved_data_chunk8
THRESHOLD=0.1

for TARGET_DATASET_PATH in /mnt/hdd2/baselines/target_data_first5_chunk8/*; do
    python experiments/libero/retrieve_libero_baseliens.py \
    --target_dataset_path $TARGET_DATASET_PATH \
    --prior_dataset_path $PRIOR_DATASET_PATH \
    --threshold $THRESHOLD \
    --batch_size 256 \
    --output_dir $OUTPUT_DIR \    
done

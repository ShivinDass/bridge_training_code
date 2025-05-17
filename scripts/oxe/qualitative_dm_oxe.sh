# DM_PATH=/home/shivin/foundation_models/deo_in_pouch_avg_dm_4_14_OXE23.npy
# OUTPUT_DIR=/mnt/hdd1/baselines/qualitative/pouch

DM_PATH=/home/shivin/foundation_models/easy_pick_avg_dm_4_7_OXE13.npy
OUTPUT_DIR=/mnt/hdd1/baselines/qualitative/ball

# DM_PATH=/home/shivin/foundation_models/tiago_sink_avg_dm_4_23_OXE24.npy
# OUTPUT_DIR=/mnt/hdd1/baselines/qualitative/tiago

# DM_PATH=/home/shivin/foundation_models/droid_avg_dm_4_30_OXE24.npy
# OUTPUT_DIR=/mnt/hdd1/baselines/qualitative/droid

TOPK=20

PRIOR_DATASET_PATH=/mnt/hdd1/baselines/embed_data

python experiments/oxe/qualitative_dm_oxe.py \
--dm_path $DM_PATH \
--prior_dataset_path $PRIOR_DATASET_PATH \
--topk $TOPK \
--batch_size 256 \
--output_dir $OUTPUT_DIR \
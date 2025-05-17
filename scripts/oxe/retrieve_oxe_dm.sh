# DM_PATH=/home/shivin/foundation_models/deo_in_pouch_avg_dm_4_14_OXE23.npy
# TOPK=0.0075

# DM_PATH=/home/shivin/foundation_models/easy_pick_avg_dm_4_7_OXE13.npy
# TOPK=0.01

# DM_PATH=/home/shivin/foundation_models/tiago_sink_avg_dm_4_23_OXE24.npy
# TOPK=0.005

DM_PATH=/home/shivin/foundation_models/droid_avg_dm_4_30_OXE24.npy
TOPK=0.01

PRIOR_DATASET_PATH=/mnt/hdd1/baselines/embed_data
OUTPUT_DIR=/mnt/hdd1/baselines/retrieved_data

python experiments/oxe/retrieve_oxe_dm.py \
--dm_path $DM_PATH \
--prior_dataset_path $PRIOR_DATASET_PATH \
--topk $TOPK \
--batch_size 256 \
--output_dir $OUTPUT_DIR \
DATA_DIR=/mnt/hdd1/tensorflow_datasets
# DATA_NAME=easy_pick_dataset_n10
# DATA_STATS_PATH="/mnt/hdd1/tensorflow_datasets/easy_pick_dataset_n10/0.1.0/dataset_statistics_fd38cefcf869387d8333828c30353151baec46175a4d169b37d61c85031ba36a.json"

# DATA_NAME=deo_in_pouch_dataset_n20
# DATA_STATS_PATH=/mnt/hdd1/tensorflow_datasets/deo_in_pouch_dataset_n20/0.1.0/dataset_statistics_dc2edbcea025e7c68dcdc2a66047d724fabfbb5d2692d86d44f1dea9a7c1341e.json

# DATA_NAME=tiago_sink_dataset_n20
# DATA_STATS_PATH=/mnt/hdd1/tensorflow_datasets/tiago_sink_dataset_n20/0.1.0/dataset_statistics_93334bb80d2fe06e5631482ffb56fa3002471e1354622e38e3683129a0988191.json

DATA_NAME=droid_dataset_multi_filtered
DATA_STATS_PATH=/mnt/hdd1/tensorflow_datasets/droid_dataset_full/0.1.0/dataset_statistics_222d447746907cf090f6ecf76fb7ac305d94d828734e293aa60e59a1cb75d194.json

### OXE STRAP
# DATA_NAME=strap_easy_pick_dataset_n10
# DATA_STATS_PATH=/mnt/hdd1/tensorflow_datasets/strap_easy_pick_dataset_n10/0.1.0/dataset_statistics_2f7b6a52e98eb50e37dd44d538128f7f69a35959a341f74ffb12bde24bfdf6a8.json
# DATA_NAME=strap_deo_in_pouch_dataset_n30
# DATA_STATS_PATH=/mnt/hdd1/tensorflow_datasets/strap_deo_in_pouch_dataset_n20/0.1.0/dataset_statistics_b39da576893b8393cf8b59e865592eaa1a4be0cf1b9e949b1a1650e020b7bbc7.json

OUT_DIR=/mnt/hdd1/baselines/target_data
echo "Computing embeddings for $DATA_NAME"
python experiments/embed_target.py \
--data_name $DATA_NAME \
--data_dir $DATA_DIR \
--batch_size 256 \
--future_image_horizon 8 \
--out_dir $OUT_DIR \
--act_pred_horizon 4 \
--data_stats_path $DATA_STATS_PATH \
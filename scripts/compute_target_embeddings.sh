DATA_DIR=/home/shivin/tensorflow_datasets
DATA_NAME=simpler_carrot_dataset
# DATA_NAME=oxe_magic_soup_s1

OUT_DIR=/mnt/hdd1/baselines/target_data


echo "Computing embeddings for $DATA_NAME"

python experiments/embed_target.py \
--data_name $DATA_NAME \
--data_dir $DATA_DIR \
--batch_size 256 \
--future_image_horizon 8 \
--out_dir $OUT_DIR
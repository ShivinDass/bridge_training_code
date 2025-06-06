DATA_DIR=/mnt/hdd1/traj_data/

OUT_DIR=/mnt/hdd1/baselines/embed_data_strap/

declare -a MIX_NAMES=(
"oxe_magic_soup_s1"
"oxe_magic_soup_s2"
"oxe_magic_soup_s3"
"oxe_magic_soup_s4"
)

MIX_NAME=${MIX_NAMES[$1]}

echo "Computing embeddings for $MIX_NAME"

python experiments/embed_oxe.py \
--data_mix $MIX_NAME \
--data_dir $DATA_DIR \
--batch_size 256 \
--future_image_horizon 8 \
--out_dir $OUT_DIR
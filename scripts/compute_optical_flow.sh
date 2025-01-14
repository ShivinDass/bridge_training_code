DATA_DIR=/mnt/hdd1/traj_data/
# OUT_DIR=/mnt/hdd1/baselines/vae_training_data/
OUT_DIR=/home/shivin/foundation_models/data/baselines/vae_training_data/

declare -a MIX_NAMES=(
"oxe_magic_soup_s1"
"oxe_magic_soup_s2"
"oxe_magic_soup_s3"
"oxe_magic_soup_s4"
)

MIX_NAME=${MIX_NAMES[$1]}

echo "Computing optical flow for $MIX_NAME"

python experiments/compute_oxe_optical_flow.py \
--data_mix $MIX_NAME \
--data_dir $DATA_DIR \
--batch_size 256 \
--horizon 8 \
--out_dir $OUT_DIR \
--out_dir_prefix $MIX_NAME \

# python experiments/compute_oxe_optical_flow.py \
# --data_mix oxe_broth \
# --data_dir $DATA_DIR \
# --batch_size 256 \
# --horizon 8 \
# --out_dir /home/shivin/foundation_models/data/baselines \
# --out_dir_prefix oxe_prechunked \
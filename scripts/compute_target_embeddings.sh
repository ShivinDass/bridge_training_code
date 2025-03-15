DATA_DIR=/home/shivin/tensorflow_datasets
DATA_NAME=libero90_horizon30
DATA_STATS_PATH=/home/shivin/tensorflow_datasets/libero90/0.1.0/dataset_statistics_9abb65a9c7829f52c81741919ae39f05baf55b6a5aab3f0ddd897947d3b283e5.json
# DATA_NAME=oxe_magic_soup_s1

OUT_DIR=/mnt/hdd2/baselines/prior_data
echo "Computing embeddings for $DATA_NAME"
python experiments/embed_target.py \
--data_name $DATA_NAME \
--data_dir $DATA_DIR \
--batch_size 256 \
--data_stats_path $DATA_STATS_PATH \
--future_image_horizon 8 \
--out_dir $OUT_DIR

# DATA_DIR=/home/shivin/tensorflow_datasets/libero_val
# OUT_DIR=/mnt/hdd2/baselines/target_data_chunk8
# DATA_STATS_PATH=/home/shivin/tensorflow_datasets/libero90/0.1.0/dataset_statistics_9abb65a9c7829f52c81741919ae39f05baf55b6a5aab3f0ddd897947d3b283e5.json

# declare -a DATASET_NAMES=(
# STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
# LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
# LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
# KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
# KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
# LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
# LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
# KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it
# KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
# LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
# )

# for i in "${!DATASET_NAMES[@]}";
# do
#     DATA_NAME="$(echo "${DATASET_NAMES[$i]}" | tr '[:upper:]' '[:lower:]')"
#     echo "Computing embeddings for ${DATA_NAME}"
#     python experiments/embed_target.py \
#         --data_name $DATA_NAME \
#         --data_dir $DATA_DIR \
#         --data_stats_path $DATA_STATS_PATH \
#         --batch_size 256 \
#         --future_image_horizon 8 \
#         --out_dir $OUT_DIR
# done

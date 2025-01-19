DATA_DIR=/home/shivin/tensorflow_datasets


python experiments/compute_oxe_optical_flow.py \
--data_mix simpler_carrot \
--data_dir $DATA_DIR \
--batch_size 256 \
--horizon 8 \
--out_dir_prefix /mnt/hdd1/simpler_carrot_flow \
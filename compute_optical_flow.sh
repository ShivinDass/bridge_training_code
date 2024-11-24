DATA_DIR=...

python experiments/compute_optical_flow.py \
--data_dir $DATA_DIR \
--batch_size 64 \
--horizon 8 \
--out_dir_prefix flow_retrieval \
set -e
declare -a task_names=(
    book-caddy_weighted
    bowl-cabinet_weighted
    moka-moka_weighted
    mug-mug_weighted
    soup-cheese_weighted
    soup-sauce_weighted
)


# DATASET=libero90_horizon30
# DATASET=libero90
DATASET=libero90_horizon15
# DATASET=libero_w15_with_subopt


PRIOR_DATASET_PATH=/mnt/hdd2/baselines/prior_data/${DATASET}_chunk8_prechunk
OUTPUT_DIR=/mnt/hdd2/baselines/retrieved_data_chunk8_dm/${DATASET}_ablation


ITERS=(25)
for i in "${!task_names[@]}"; do
    for j in "${!ITERS[@]}"; do
        TASK=${task_names[$i]}
        DM_PATH=/mnt/hdd2/dm_experiments/${DATASET}_128x128/${TASK}
        ITER=${ITERS[$j]}

        python experiments/libero/retrieve_libero_dm.py \
        --dm_path $DM_PATH \
        --prior_dataset_path $PRIOR_DATASET_PATH \
        --batch_size 256 \
        --output_dir $OUTPUT_DIR \
        --topk 0.1 \
        --iter_dm $ITER \
    
    done
done
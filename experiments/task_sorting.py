import argparse
import tqdm
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from jaxrl_m.data.custom_retrieval_dataset import CustomRetrievalDataset

def print_nested_dict(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print('\t' * indent, k)
            print_nested_dict(v, indent + 1)
        else:
            try:
                print('\t' * indent, k, v.shape)
            except:
                pass


def main():
        
    # dataset_path = '/mnt/hdd1/baselines/target_data/easy_pick_dataset_n10_h8_prechunk/train/out.tfrecord'
    # dataset_path = '/mnt/hdd1/baselines/retrieved_data_2/easy_pick_dataset_n10_h8_prechunk_th0.005/flow/out.tfrecord'
    dataset_path = '/mnt/hdd1/baselines/embed_data/oxe_magic_soup_s1_h8_prechunk/out.tfrecord'
    # dataset_path = f'/mnt/hdd1/baselines/retrieved_data/{dataset_name}/{retrieval_method}/out.tfrecord'
    ds = CustomRetrievalDataset(
        [[dataset_path]],
        seed=0,
        train=True,
        batch_size=256,
        load_keys=['observation/image_primary', 'dataset_name', 'task/language_instruction'],
        decode_imgs=True
    )
    iter = ds.iterator()

    count = 0
    pbar = tqdm.tqdm(total=None)
    for i, batch in enumerate(iter):
        # print(batch['observation/image_primary'][0])
        # print(batch['dataset_name'][0])
        # print_nested_dict(batch)

        for instruction in batch['task/language_instruction']:
            instruction = instruction.decode()
            if 'put carrot' in instruction:# and 'plate' in instruction:
            # if 'eggplant' in instruction and 'yellow basket' in instruction:
            # if 'spoon' in instruction and 'towel' in instruction:
            # if 'stack' in instruction and 'block' in instruction:
                count += 1
                print(instruction)
        pbar.update(1)

    print(count)

if __name__=='__main__':
    main()
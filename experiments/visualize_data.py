import argparse
import tqdm
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb
import datetime
from jaxrl_m.data.octo_dataset import BaselinesOctoDataset
from jaxrl_m.data.custom_retrieval_dataset import CustomRetrievalDataset

IMAGES_TO_SAVE=5

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


def visualize_data(dataset_name, retrieval_method):
    WANDB_ENTITY = 'ut-robin'
    WANDB_PROJECT = 'vis_rlds'

    if WANDB_ENTITY is not None:
        render_wandb = True
        wandb.init(entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                id=f"{retrieval_method}_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
    else:
        render_wandb = False
        
    # dataset_path = '/mnt/hdd1/baselines/target_data/easy_pick_dataset_n10_h8_prechunk/train/out.tfrecord'
    # dataset_path = '/mnt/hdd1/baselines/retrieved_data_2/easy_pick_dataset_n10_h8_prechunk_th0.005/flow/out.tfrecord'
    # dataset_path = '/mnt/hdd1/baselines/embed_data_2/oxe_magic_soup_s1_h8_prechunk/out.tfrecord'
    dataset_path = f'/mnt/hdd1/baselines/retrieved_data/{dataset_name}/{retrieval_method}/out.tfrecord'
    ds = CustomRetrievalDataset(
        [[dataset_path]],
        seed=0,
        train=True,
        batch_size=256,
        load_keys=['observation/image_primary', 'observation/image_wrist', 'dataset_name'],
        decode_imgs=True
    )
    ds.tf_dataset = ds.tf_dataset.shuffle(10000)
    iter = ds.iterator()

    dataset_names = {}
    for i, batch in enumerate(iter):
        # print(batch['observation/image_primary'][0])
        # print(batch['dataset_name'][0])
        # print_nested_dict(batch)
        # exit(0)
        if i < IMAGES_TO_SAVE:
            image_strip_primary = np.concatenate(batch['observation/image_primary'][:18], axis=1)
            image_strip_wrist = np.concatenate(batch['observation/image_wrist'][:18], axis=1)

            image_strip = np.concatenate([image_strip_primary, image_strip_wrist], axis=0)
            image_strip = np.concatenate([image_strip[:, :256*6], image_strip[:, 256*6:256*12], image_strip[:, 256*12:]], axis=0)
            
            if render_wandb:
                wandb.log({f'image_{i}': wandb.Image(image_strip)})
            else:
                plt.figure()
                plt.imshow(image_strip)
                # plt.title(caption)
                plt.savefig(f'hi_qual_episode_{i}.png')

        for name in batch['dataset_name']:
            name = name.decode()
            if name not in dataset_names:
                dataset_names[name] = 0
            dataset_names[name] += 1

    # create a bar plot of dataset names and their frequencies sorted by frequency
    dataset_names = {k[:10]: v for k, v in sorted(dataset_names.items(), key=lambda item: item[1], reverse=True)}
    fig = plt.figure()
    plt.bar(dataset_names.keys(), dataset_names.values())
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.title('Dataset name frequencies')
    plt.tight_layout()
    if render_wandb:
        wandb.log({'dataset_name_frequencies': wandb.Image(fig)})
    else:
        plt.savefig('dataset_frequencies.png')

    wandb.finish()

if __name__=='__main__':
    dataset_name = 'simpler_carrot_dataset_h8_prechunk_th0.05'
    retrieval_methods = ['flow', 'br', 'action']
    for retrieval_method in retrieval_methods:
        visualize_data(dataset_name, retrieval_method)
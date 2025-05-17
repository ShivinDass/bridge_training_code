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
from jaxrl_m.data.custom_retrieval_dataset import CustomRetrievalDataset

ITER=30
TOP=0.05

TASK = 'book-caddy'
# TASK = 'bowl-cabinet'
# TASK='mug-mug'
# TASK='moka-moka'
# TASK='cream-butter'
# TASK='soup-sauce'

# DATASET = 'libero90_horizon30'
DATASET='libero_w15_with_subopt'
# DATASET = 'libero90'

relevant_task_map ={
    'book-caddy': [
            'pick up the book and place it in the left compartment of the caddy',
            'pick up the book and place it in the right compartment of the caddy',
            'pick up the book and place it in the front compartment of the caddy',
            'pick up the book and place it in the back compartment of the caddy',
        ],
    'bowl-cabinet': [
            'close the bottom drawer of the cabinet',
            'close the bottom drawer of the cabinet and open the top drawer',
            'put the black bowl in the bottom drawer of the cabinet',
            'put the black bowl on top of the cabinet',
        ],
    'mug-mug': [
        'put the white mug on the left plate',
        'put the yellow and white mug on the right plate'
    ],
    'moka-moka': [
        'put the moka pot on the stove',
        'put the right moka pot on the stove',
    ],
    "cream-butter" : [
        'pick up the cream cheese and put it in the tray',
        'pick up the chocolate pudding and put it in the tray',
        'pick up the butter and put it in the basket',
    ],
    "soup-sauce": [
    ]
}

IMAGES_TO_SAVE=0
IMG_SIZE = 128

def load_all_libero90_tasks():
    path = '/mnt/hdd2/libero/libero_90'
    name2domain_and_instruction = {}
    for task in os.listdir(path):
        task = task.split('.')[0]

        language_instruction = task.split('SCENE')[1][2:]
        language_instruction = " ".join(language_instruction.split('_')[:-1])
        domain = task[:-len(language_instruction)-6]
        name2domain_and_instruction[task] = (domain, language_instruction)
    return name2domain_and_instruction


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


def visualize_data(task_name, retrieval_method):
    WANDB_ENTITY = 'ut-robin'
    WANDB_PROJECT = 'vis_rlds'

    if retrieval_method=='dm':
        # dataset_path = '/mnt/hdd2/baselines/prior_data/libero90_horizon30_chunk8_prechunk/out.tfrecord'
        dataset_path = f'/mnt/hdd2/baselines/retrieved_data_chunk8_dm/{DATASET}/{TASK}_weighted_subopt_iter{ITER}_top{TOP}.tfrecord'
        wandb_name = f"{retrieval_method}_{DATASET}_{os.path.basename(dataset_path)[:-9]}"
    else:
        base_path = '/mnt/hdd2/baselines/with_subopt_embeddings_retrieved_data_chunk8'
        if TASK=='book-caddy':
            dataset_path = f'{base_path}/study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_chunk8_prechunk/study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_chunk8_prechunk_th{TOP}/{retrieval_method}/out.tfrecord'
        elif TASK=='bowl-cabinet':
            dataset_path = f'{base_path}/kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_chunk8_prechunk/kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_chunk8_prechunk_th{TOP}/{retrieval_method}/out.tfrecord'
        elif TASK=='mug-mug':
            dataset_path = f'{base_path}/living_room_scene5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_chunk8_prechunk/living_room_scene5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_chunk8_prechunk_th{TOP}/{retrieval_method}/out.tfrecord'
        elif TASK=='moka-moka':
            dataset_path = f'{base_path}/kitchen_scene8_put_both_moka_pots_on_the_stove_chunk8_prechunk/kitchen_scene8_put_both_moka_pots_on_the_stove_chunk8_prechunk_th{TOP}/{retrieval_method}/out.tfrecord'
        elif TASK=='cream-butter':
            dataset_path = f'{base_path}/living_room_scene2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_chunk8_prechunk/living_room_scene2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_chunk8_prechunk_th{TOP}/{retrieval_method}/out.tfrecord'
        elif TASK=='soup-sauce':
            dataset_path = f'{base_path}/living_room_scene2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_chunk8_prechunk/living_room_scene2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_chunk8_prechunk_th{TOP}/{retrieval_method}/out.tfrecord'
        else:
            raise ValueError(f'Unknown task {TASK}')
        wandb_name = f"{retrieval_method}_{DATASET}_{TASK}"
    
    if WANDB_ENTITY is not None:
        render_wandb = True
        wandb.init(entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                id=f"{wandb_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
    else:
        render_wandb = False
        
    # dataset_path = '/mnt/hdd2/baselines/prior_data/libero90_horizon15_chunk8_prechunk/out.tfrecord'
    # dataset_path2 = '/mnt/hdd2/baselines/prior_data/libero90_subopt_horizon15_chunk8_prechunk/out.tfrecord'
    # dataset_path = '/mnt/hdd2/baselines/target_data/kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_chunk8_prechunk/val/out.tfrecord'
    
    ds = CustomRetrievalDataset(
        [[dataset_path]],
        # [[dataset_path], [dataset_path2]],
        train=True,
        batch_size=512,
        load_keys=['observation/image_primary', 'observation/image_wrist', 'dataset_name', 
                   'task/language_instruction', 'index', 'action', 'is_suboptimal'],
        decode_imgs=False
    )
    iter = ds.iterator()

    dataset_names = {}
    language_instructions = {}
    indexes = set({})
    actions = []
    instruction_to_idx = {}
    is_suboptimal = {}
    for j, batch in enumerate(iter):
        if j < IMAGES_TO_SAVE:
            image_strip_primary = np.concatenate(batch['observation/image_primary'][:18], axis=1)
            image_strip_wrist = np.concatenate(batch['observation/image_wrist'][:18], axis=1)

            print(image_strip_primary.shape)
            print(image_strip_wrist.shape)

            image_strip = np.concatenate([image_strip_primary, image_strip_wrist], axis=0)
            image_strip = np.concatenate([image_strip[:, :IMG_SIZE*6], image_strip[:, IMG_SIZE*6:IMG_SIZE*12], image_strip[:, IMG_SIZE*12:]], axis=0)
            
            if render_wandb:
                wandb.log({f'image_{i}': wandb.Image(image_strip)})
            else:
                plt.figure()
                plt.imshow(image_strip)
                plt.savefig(f'hi_qual_episode_{i}.png')

        # for name in batch['dataset_name']:
        #     name = name.decode()
        #     if name not in dataset_names:
        #         dataset_names[name] = 0
        #     dataset_names[name] += 1

        for i, instruction in enumerate(batch['task/language_instruction']):
            # if batch['index'][i][0] in indexes:
            #     continue
            instruction = instruction.decode()
            if instruction not in language_instructions:
                language_instructions[instruction] = 0
                is_suboptimal[instruction] = 0
            language_instructions[instruction] += 1

            is_suboptimal[instruction] += batch['is_suboptimal'][i]
            assert (not batch['index'][i][0]>46705) or batch['is_suboptimal'][i], "Suboptimal action should be 1 if index is > 0"

            if instruction not in instruction_to_idx:
                instruction_to_idx[instruction] = set({})
            instruction_to_idx[instruction].add(batch['index'][i][0])

            indexes.add(batch['index'][i][0])

        actions.append(batch['action'])
        if (j+1) % 500 == 0:
            print(f"Processed {j+1} batches")

    # import pickle
    # with open('/home/shivin/libero_experiments/instruction_to_idx_w15_with_subopt.pkl', 'wb') as f:
    #     pickle.dump(instruction_to_idx, f)
    
    # create a bar plot of dataset names and their frequencies sorted by frequency
    # dataset_names = {k[:10]: v for k, v in sorted(dataset_names.items(), key=lambda item: item[1], reverse=True)}
    
    # name2domain_and_instruction = load_all_libero90_tasks()

    # fig = plt.figure()
    # domain_frequencies = {k: v for k, v in sorted(domain_frequencies.items(), key=lambda item: item[1], reverse=True)}
    # plt.bar(domain_frequencies.keys(), domain_frequencies.values())
    # plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    # plt.title('Domain frequencies')
    # plt.tight_layout()
    # if render_wandb:
    #     wandb.log({'domain_frequencies': wandb.Image(fig)})
    # else:
    #     plt.savefig('domain_frequencies.png')
    
    total = np.sum(list(language_instructions.values()))
    print(total)
    language_instructions = {k: v/total for k, v in sorted(language_instructions.items(), key=lambda item: item[1], reverse=True)}
    
    fig = plt.figure(figsize=(20, 10))

    keys = list(language_instructions.keys())
    is_suboptimal = [is_suboptimal[k]/total for k in keys]
    colors = ['C0']*len(keys)
    for k in keys:
        if k in relevant_task_map[TASK]: 
            colors[keys.index(k)] = 'C1'
    plt.bar(language_instructions.keys(), language_instructions.values(), color=colors)
    plt.bar(language_instructions.keys(), is_suboptimal, color='red')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.title(f'Language frequency - {TASK} ({retrieval_method})')
    plt.ylim(0, 0.3)
    plt.tight_layout()
    if render_wandb:
        wandb.log({'language_instruction_frequencies': wandb.Image(fig)})
    else:
        plt.savefig('language_instruction_frequencies.png')

    print('Unique indexes', np.unique(list(indexes)).shape[0])

    actions = np.concatenate(actions, axis=0)[:, 0]
    print(np.mean(actions, axis=0), np.std(actions, axis=0))
    print(np.unique(actions[:, -1]))
    wandb.finish()

if __name__=='__main__':
    # dataset_name = 'simpler_carrot_dataset_chunk8_prechunk_th0.05'
    # dataset_name = 'kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_chunk8_prechunk_th0.1'
    retrieval_methods = ['flow', 'br', 'action']
    for retrieval_method in retrieval_methods:
        visualize_data(f'{TASK}_baseline', retrieval_method)
    # visualize_data(task_name=f'{TASK}', retrieval_method="dm")
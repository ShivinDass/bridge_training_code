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

IMAGES_TO_SAVE=0
IMG_SIZE = 256

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


def get_all_dataset_keys(path):
    dataset = tf.data.TFRecordDataset(path, num_parallel_reads=tf.data.AUTOTUNE)
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        all_dataset_keys = list(example.features.feature.keys())

    return all_dataset_keys

def visualize_data(task_name, retrieval_method):
    WANDB_ENTITY = None #'ut-robin'
    WANDB_PROJECT = None #'vis_rlds'
    
    if WANDB_ENTITY is not None:
        render_wandb = True
        wandb.init(entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                id=f"real_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
    else:
        render_wandb = False
        
    # dataset_path = '/mnt/hdd1/baselines/retrieved_data/dm_ut_oxe_easy_pick_n10/out.tfrecord'
    # dataset_path = '/mnt/hdd1/baselines/embed_data_2/ut_datasets_h8_prechunk/out.tfrecord'
    # dataset_path = '/mnt/hdd1/baselines/retrieved_data/oxe23_deo_in_pouch/oxe_23_prechunk_top0.0075/out.tfrecord'
    
    # baselines
    # dataset_path = '/mnt/hdd1/baselines/retrieved_data/deo_in_pouch_dataset_n20_chunk4_prechunk_th0.0075/br/out.tfrecord'
    # dataset_path = '/mnt/hdd1/baselines/retrieved_data/tiago_sink_dataset_n20_chunk4_prechunk_th0.005/strap/out.tfrecord'
    # dataset_path = '/mnt/hdd1/baselines/retrieved_data/deo_in_pouch_avg_dm_4_17_OXE24_top0.0025.tfrecord'

    # dataset_path = f'/mnt/hdd1/baselines/retrieved_data/tiago_sink_dataset_n20_chunk4_prechunk_th0.005/{retrieval_method}/out.tfrecord'
    # out_prefix =  f'tiago_{retrieval_method}'
    
    dataset_path = f'/mnt/hdd1/baselines/retrieved_data/oxe23_deo_in_pouch/deo_in_pouch_dataset_n20_chunk4_prechunk_th0.0075/{retrieval_method}/out.tfrecord'
    out_prefix =  f'franka_deo_{retrieval_method}'

    ds = CustomRetrievalDataset(
        [[dataset_path]],
        seed=0,
        train=True,
        batch_size=256,
        # load_keys=['observation/image_primary', 'observation/image_wrist', 'dataset_name', 'task/language_instruction', 'index', 'action'],
        load_keys=['dataset_name', 'task/language_instruction', 'index', 'action'],
        decode_imgs=True
    )
    # ds.tf_dataset = ds.tf_dataset.shuffle(buffer_size=10000)
    iter = ds.iterator()

    language_instructions = {}
    dataset_names = {}
    indexes = set({})
    actions = []
    for i, batch in enumerate(iter):
        if i < IMAGES_TO_SAVE:
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

        for name in batch['dataset_name']:
            name = name.decode()
            if name not in dataset_names:
                dataset_names[name] = 0
            dataset_names[name] += 1

        for i, instruction in enumerate(batch['task/language_instruction']):
            # if batch['index'][i][0] in indexes:
            #     continue
            instruction = instruction.decode()
            if instruction not in language_instructions:
                language_instructions[instruction] = 0
            language_instructions[instruction] += 1

            indexes.add(batch['index'][i])

        actions.append(batch['action'])

    # import pickle
    # with open('/home/shivin/libero_experiments/instruction_to_idx.pkl', 'wb') as f:
    #     pickle.dump(instruction_to_idx, f)

    print('Unique indexes', np.unique(list(indexes)).shape[0])
    
    # plot dataset name distribution
    # plt.figure()
    # dataset_names = {k: v for k, v in sorted(dataset_names.items(), key=lambda item: item[1], reverse=True)}
    # keys = list(dataset_names.keys())
    # plt.bar(range(len(keys)), dataset_names.values(), tick_label=[remap_dataset_names[k] for k in keys])
    # plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    # plt.ylabel('Frequency')
    # plt.title(retrieval_method)
    # plt.tight_layout()
    # plt.savefig(f'/home/shivin/foundation_models/z_visuals/{out_prefix}_dataset_distribution.png')

    total = np.sum(list(dataset_names.values()))
    dataset_names = {k: v/total for k, v in dataset_names.items()}
    dataset_names = {k: v for k, v in sorted(dataset_names.items(), key=lambda item: item[1], reverse=True)}
    keys = list(dataset_names.keys())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # ax.bar(range(len(keys)), dataset_names.values())
    # ax.set_xticks(range(len(keys)))
    # ax.set_xticklabels([remap_dataset_names[k] for k in keys], rotation=45, ha='right', rotation_mode='anchor')
    # ax.set_ylabel('Dataset Composition of Selected Data') 
    # ax.set_yticks(np.arange(0, np.max(list(dataset_names.values()))+0.1, 0.1))
    # ax.spines[['top', 'right']].set_visible(False)

    ax.pie(dataset_names.values(), \
           labels=[remap_dataset_names[k] if dataset_names[k]>0.02 else '' for k in keys], startangle=90, counterclock=False, colors=plt.cm.tab20.colors)
    # plt.tight_layout()
    plt.savefig(f'/home/shivin/foundation_models/z_visuals/{out_prefix}_dataset_distribution.png')
    plt.savefig(f'/home/shivin/foundation_models/z_visuals/{out_prefix}_dataset_distribution.pdf', bbox_inches='tight', dpi=300)

    return dataset_names

remap_dataset_names = dict(
    fractal20220817_data='RT1',
    kuka='Kuka',
    bridge_dataset='Bridge',
    taco_play='Taco Play',
    jaco_play='Jaco Play',
    berkeley_cable_routing='Berkeley Cable',
    roboturk='Roboturk',
    nyu_door_opening_surprising_effectiveness='NYU Door Opening',
    viola='Viola',
    berkeley_autolab_ur5='Berkeley Autolab UR5',
    toto='Toto',
    stanford_hydra_dataset_converted_externally_to_rlds='Stanford Hydra',
    austin_buds_dataset_converted_externally_to_rlds='Austin Buds',
    nyu_franka_play_dataset_converted_externally_to_rlds='NYU Franka',
    furniture_bench_dataset_converted_externally_to_rlds='Furniture\nBench',
    ucsd_kitchen_dataset_converted_externally_to_rlds='UCSD Kitchen',
    austin_sailor_dataset_converted_externally_to_rlds='Austin Sailor',
    austin_sirius_dataset_converted_externally_to_rlds='Austin Sirius',
    bc_z='BC-Z',
    dlr_edan_shared_control_converted_externally_to_rlds='DLR Edan',
    iamlab_cmu_pickup_insert_converted_externally_to_rlds='CMU IAM\nLab',
    utaustin_mutex='Austin Mutex',
    berkeley_fanuc_manipulation='Berkeley\nFanuc',
    cmu_stretch='CMU Stretch'
)

if __name__=='__main__':
    # dataset_name = 'simpler_carrot_dataset_h8_prechunk_th0.05'
    # dataset_name = 'kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_h8_prechunk_th0.1'
    # retrieval_methods = ['dm']#, 'flow', 'br', 'action', 'strap'] #'random'
    retrieval_methods = ['dm','br','action', 'flow']
    # retrieval_methods = ['br', 'dm']
    aggregate_datanames = {}
    all_datasets = set()
    for retrieval_method in retrieval_methods:
        dataset_names = visualize_data(f'', retrieval_method)
        aggregate_datanames[retrieval_method] = dataset_names
        all_datasets.update(dataset_names.keys())

    exit(0)
    # visualize_data(task_name='', retrieval_method="dm")

    def gen_all_dataset_dist_plit():
        def normalize_dict(d):
            total = sum(d.values())
            for k in d:
                d[k] /= total
            return d
        aggregate_datanames = {k: normalize_dict(v) for k, v in aggregate_datanames.items()}

        width = 0.9/len(retrieval_methods)

        keys = list(aggregate_datanames['dm'].keys())[:8]
        # for dataset in all_datasets:
        #     if dataset not in keys:
        #         keys.append(dataset)

        fig, ax = plt.subplots(figsize=(10, 5))
        # ax.bar(np.arange(len(keys))+0.5, [int(i%2) for i, _ in enumerate(keys)], width=1, color=(0.9,0.9,0.9))

        for i, retrieval_method in enumerate(retrieval_methods):
            dataset_names = aggregate_datanames[retrieval_method]
            dataset_names = {k: dataset_names.get(k, 0) for k in keys}
            
            ax.bar(np.arange(len(keys)) + i*width, dataset_names.values(), width=width, label=retrieval_method)

        
        ax.set_ylim(-0.05, 0.5)
        ax.set_xticks(np.arange(len(keys)) + 0.5)
        ax.set_xticklabels([remap_dataset_names[k] for k in keys])#, rotation=45, ha='right', rotation_mode='anchor')
        ax.legend()
        plt.ylabel('Frequency')
        plt.title('Dataset Distribution')
        plt.tight_layout()
        plt.savefig(f'/home/shivin/foundation_models/z_visuals/all_dataset_distribution.png')
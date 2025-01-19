from jaxrl_m.data.retrieval_dataset import RetrievalDataset

import tqdm
import numpy as np

# paths = '/mnt/hdd1/bridge_dataset_flow_h8_prechunk/train/out.tfrecord'
paths = '/mnt/hdd1/retrieved_simpler_carrot/retrieved_simpler_carrot_flow_retrieved_0.01_prechunk/train/out.tfrecord'
# paths = '/mnt/hdd1/simpler_carrot_flow_h8_prechunk/train/out.tfrecord'

data = RetrievalDataset(
        [[paths]],
        batch_size=256,
        act_pred_horizon=8,
        prechunk=True,
        flow_dtype='float16',
    )
data_iter = data.iterator()

all_actions = []
pbar = tqdm.tqdm(total=None)
while True:
    try:
        batch = next(data_iter)
        all_actions.append(batch['actions'])
    except StopIteration:
        break
    pbar.update(1)
pbar.close()
all_actions = np.concatenate(all_actions, axis=0).reshape(-1, 7)

print(np.mean(all_actions, axis=0))
print(np.std(all_actions, axis=0))

all_grasps = all_actions[:, -1]
print()
close1 = np.sum(np.isclose(all_grasps, 1.0))
close2 = np.sum(np.isclose(all_grasps, 0.0))
print(close1, close2)
print(close1 + close2)
print(all_grasps.shape)

print(all_actions[-1])
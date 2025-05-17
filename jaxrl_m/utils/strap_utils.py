import numpy as np
from itertools import accumulate
import numba as nb

### Code adapted from STRAP: https://github.com/WEIRDLabUW/STRAP

def segment_trajectory_by_derivative(states, threshold=2.5e-3):
    # Calculate the absolute derivative of the first three states (X, Y, Z)
    diff = np.diff(states[:, :3], axis=0)
    abs_diff = np.sum(np.abs(diff), axis=1)

    # Find points where the derivative is below the threshold (indicating a stop)
    stops = np.where(abs_diff < threshold)[0]

    # Initialize the sub-trajectories list
    sub_trajectories = []
    start_idx = 0

    # Segment the trajectory at each stop point
    for stop in stops:
        sub_trajectories.append(states[start_idx : stop + 1])  # Add the segment
        start_idx = stop + 1  # Update start index

    # Append the last remaining segment
    if start_idx < len(states):
        sub_trajectories.append(states[start_idx:])

    return sub_trajectories


def merge_short_segments(segments, min_length=5):
    merged_segments = []
    current_segment = segments[0]

    for i in range(1, len(segments)):
        # If the current segment is too short, merge it with the next
        if len(current_segment) < min_length:
            current_segment = np.vstack((current_segment, segments[i]))
        else:
            merged_segments.append(
                current_segment
            )  # Save the segment if it's long enough
            current_segment = segments[i]  # Start a new segment

        prev_segment = current_segment

    # If the last segment is too short, merge it with the previous
    if len(current_segment) < min_length:
        merged_segments[-1] = np.vstack((merged_segments[-1], current_segment))
    else:
        merged_segments.append(current_segment)

    return merged_segments

def slice_embeddings(eef_poses, task_embeddings, min_length=40):
    new_task_embeddings = []
    for (eef_pose, task_embedding) in zip(eef_poses, task_embeddings):
        # segment using state derivative heuristic
        segments = segment_trajectory_by_derivative(
            eef_pose, threshold=5e-2
        )
        merged_segments = merge_short_segments(
            segments, min_length=min_length
        )

        # extract slice indexes
        seg_idcs = [0] + list(accumulate(len(seg) for seg in merged_segments))

        for i in range(len(seg_idcs) - 1):
            new_task_embeddings.append(task_embedding[seg_idcs[i] : seg_idcs[i + 1]])
    del task_embeddings
    return new_task_embeddings

from transformers import Dinov2Model, AutoImageProcessor
import torch
class DinoV2:
    def __init__(self, device):
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
        self.device = device
        self.pooling = 'avg'
    
    def preprocess(self, imgs):
        inputs = self.processor(images=imgs, return_tensors="pt")

        return inputs["pixel_values"].to(self.device)
    
    def encode(self, postprocessed_imgs):

        outputs = self.model(pixel_values=postprocessed_imgs, output_hidden_states=True)

        features = outputs.last_hidden_state

        if self.pooling is not None:
            if self.pooling == "avg":
                features = torch.mean(features, dim=1)
            elif self.pooling == "max":
                features = torch.max(features, dim=1).values

        # elif self.token_idx is not None:
        #     # [cls] token of last layer -> self.token_idx = 0
        #     # https://github.com/huggingface/transformers/blob/1f9f57ab4c8c30964360a2ba697c339f6d31f03f/src/transformers/models/dinov2/modeling_dinov2.py#L711
        #     features = features[:, self.token_idx]

        return features


# DTW
def get_distance_matrix(sub_traj_emb, prior_emb):
    sub_squared = np.sum(sub_traj_emb**2, axis=1)[:, np.newaxis]
    dataset_squared = np.sum(prior_emb**2, axis=1)[:, np.newaxis]

    cross_term = np.dot(sub_traj_emb, prior_emb.T)
    # since ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a * b
    distance_matrix = np.sqrt(sub_squared - 2 * cross_term + dataset_squared.T)

    return distance_matrix

# Most of this code was taken from this reference https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html
@nb.jit(nopython=True)
def compute_accumulated_cost_matrix_subsequence_dtw_21(C):
    """
    Args:
        C (np.ndarray): Cost matrix
    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N + 1, M + 2))
    D[0:1, :] = np.inf
    D[:, 0:2] = np.inf

    D[1, 2:] = C[0, :]

    for n in range(1, N):
        for m in range(0, M):
            if n == 0 and m == 0:
                continue
            D[n + 1, m + 2] = C[n, m] + min(
                D[n - 1 + 1, m - 1 + 2], D[n - 1 + 1, m - 2 + 2]
            )  # D[n-2+1, m-1+2],
    D = D[1:, 2:]
    return D

@nb.jit(nopython=True)
def compute_optimal_warping_path_subsequence_dtw_21(D, m=-1):
    """
    Args:
        D (np.ndarray): Accumulated cost matrix
        m (int): Index to start back tracking; if set to -1, optimal m is used (Default value = -1)

    Returns:
        P (np.ndarray): Optimal warping path (array of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n - 1, m - 1], D[n - 1, m - 2])  # D[n-2, m-1],
            if val == D[n - 1, m - 1]:
                cell = (n - 1, m - 1)
            # elif val == D[n-2, m-1]:
            #     cell = (n-2, m-1)
            else:
                cell = (n - 1, m - 2)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P
    

class SubsequenceDTW:

    def __init__(self, start, end, cost, traj_index):
        self.start = start
        self.end = end
        self.cost = cost
        self.traj_index = traj_index
    
    def __ge__(self, other):
        return self.cost >= other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __gt__(self, other):
        return self.cost > other.cost
    
    def __repr__(self):
        return f"SubsequenceDTW(start={self.start}, end={self.end}, cost={self.cost})"
    
    def __len__(self):
        return self.end - self.start

def get_single_match(sub_traj_emb, prior_emb, traj_index):
    if len(sub_traj_emb) > len(prior_emb):
        # print(len(prior_emb))
        return None
    
    distance_matrix = get_distance_matrix(sub_traj_emb, prior_emb)
    accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_21(distance_matrix)
    path = compute_optimal_warping_path_subsequence_dtw_21(accumulated_cost_matrix)

    start = path[0, 1]
    if start < 0:
        assert start == -1
        start = 0
    end = path[-1, 1]
    cost = accumulated_cost_matrix[-1, end]
    # Note that the actual end index is inclusive in this case so +1 to use python : based indexing
    end = end + 1

    return SubsequenceDTW(start, end, cost, traj_index)

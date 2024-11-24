## How to run the code?

1. Check `compute_optical_flow.sh` for computing optical flow for your dataset.
2. Check `pretrain.sh` for pretraining optical flow VAE.
3. Check `retrieve.sh` for retrieving relevant data.
4. Check `train.sh` for flow-guided training with target and retrieved data.

## Cite

This codebase is based on [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2).

If you use this code and/or FlowRetrieval in your work, please cite the paper with:

```
@misc{lin2024flowretrieval,
      title={FlowRetrieval: Flow-Guided Data Retrieval for Few-Shot Imitation Learning}, 
      author={Li-Heng Lin and Yuchen Cui and Amber Xie and Tianyu Hua and Dorsa Sadigh},
      year={2024},
      eprint={2408.16944},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2408.16944}, 
}
```
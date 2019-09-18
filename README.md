# Do Multi-hop Readers Dream of Reasoning Chains?

## Notes:
The codes are modified based on the original BERT [repo](https://github.com/google-research/bert). Some unrelated modules from the original repo have been deleted in order to make it easy to understand.

## Data:
We provide the processed data mentioned in the paper under `data/`.

### Training
Please refer to run.sh for the entry point

### Inference
Please refer to run.sh for the entry point

### Results
```
random: 75.27%
single_oracle: 77.86%
ordered_oracle: 79.15%
ordered_oracle_with_token: 79.49%
oredered_oracle_co_matching: 80.77%
```

to evaluate using the script: python evaluate-v1.1.py data/hotpot_dev_distractor_v1.ordered_support_only.json <model_folder>/prediction.json

# BeaconNet: Finding clues and solutions for safety alignment loss during pruning & quantization

## Repo structure

1. data: mid results during research

2. datasets & models: materials used for research

3. src: codes for activation analysis, model pre-alignment, evaluation and pruning

4. scripts: convenient bash scripts for running the research codes

5. config.py: all configurations (hyperparameters, paths and other settings) used in the research

## Methodology (current)

Use Sparse Autoencoders (SAEs) to recognize critical parameters and features for safety alignment, patch these features for causal interpretation.
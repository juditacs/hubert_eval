Code and result tables for the ["Evaluating Contextualized Language Models for Hungarian"](https://arxiv.org/abs/2102.10848) MSZNY2021 paper

# Prerequisites

The source code for running the experiments is available in the [probing](https://github.com/juditacs/probing) package.
This repository only contains the configuration files, the results tables and the analysis notebook.

# Running all experiments

Run all morphology experiments (6th layer only):

    python $PROBING_PATH/train_many_configs.py -c config/common.yaml -p config/generate_all.py

# Cite

```
@InProceedings{   Acs:2021,
  author        = {Ács, Judit and L\'evai, D\'aniel and Nemeskey, D\'avid M\'ark and Kornai, András},
  title         = {Evaluating Contextualized Language Models for Hungarian},
  booktitle     = {{XVII}.\ Magyar Sz{\'a}m{\'i}t{\'o}g{\'e}pes Nyelv{\'e}szeti Konferencia ({MSZNY}2020)},
  year          = 2021,
  address       = {Szeged}
}
```

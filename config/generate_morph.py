#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import logging
import os
import gc
import torch

from probing.config import Config


def generate_configs(config_fn):
    this_dir = os.path.dirname(__file__)
    data_root = os.path.join(this_dir, "..", "data")
    morph_basedir = os.path.join(data_root, "morph")
    models = {
        'SZTAKI-HLT/hubert-base-cc': 12,
        "xlm-mlm-100-1280": 16,
        "bert-base-multilingual-cased": 12,
        "xlm-roberta-base": 12,
        "distilbert-base-multilingual-cased": 6,
    }
    for model, layer_num in models.items():
        for layer in list(range(layer_num + 1)) + ['weighted_sum']:
            for task_path in os.scandir(morph_basedir):
                task = task_path.name
                for subword in ['first', 'last']:
                    logging.info("=====================================")
                    logging.info(f"=== {model} {layer} {task} {subword} ===")
                    logging.info("=====================================")
                    train_file = f"{task_path.path}/train.tsv"
                    dev_file = f"{task_path.path}/dev.tsv"
                    config = Config.from_yaml(config_fn)
                    config.model = 'SentenceRepresentationProber'
                    config.dataset_class = 'SentenceProberDataset'
                    config.subword_pooling = subword
                    config.layer_pooling = layer
                    config.model_name = model
                    config.train_file = train_file
                    config.dev_file = dev_file
                    yield config
                    gc.collect()
                    torch.cuda.empty_cache()
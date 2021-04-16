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
    szeged_pos_train = os.path.join(data_root, "szeged_ud_pos", "train")
    szeged_pos_dev = os.path.join(data_root, "szeged_ud_pos", "dev")
    full_pos_train = os.path.join(data_root, "pos", "train")
    full_pos_dev = os.path.join(data_root, "pos", "dev")
    models = {
        'SZTAKI-HLT/hubert-base-cc': 12,
        "xlm-mlm-100-1280": 16,
        "bert-base-multilingual-cased": 12,
        "xlm-roberta-base": 12,
        "distilbert-base-multilingual-cased": 6,
    }
    for model, layer_num in models.items():
        for layer in list(range(layer_num + 1)) + ['weighted_sum']:
            for subword in ['first', 'last']:
                logging.info("=====================================")
                logging.info(f"=== {model} {layer} {subword} ===")
                logging.info("=====================================")
                config = Config.from_yaml(config_fn)
                config.model = 'TransformerForSequenceTagging'
                config.dataset_class = 'SequenceClassificationWithSubwords'
                config.subword_pooling = subword
                config.layer_pooling = layer
                config.model_name = model
                config.train_file = szeged_pos_train
                config.dev_file = szeged_pos_dev
                yield config
                gc.collect()
                torch.cuda.empty_cache()

                config = Config.from_yaml(config_fn)
                config.model = 'TransformerForSequenceTagging'
                config.dataset_class = 'SequenceClassificationWithSubwords'
                config.subword_pooling = subword
                config.layer_pooling = layer
                config.model_name = model
                config.train_file = full_pos_train
                config.dev_file = full_pos_dev
                yield config
                gc.collect()
                torch.cuda.empty_cache()
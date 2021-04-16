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
    szeged_ner_train = os.path.join(data_root, "szeged_ner", "train")
    szeged_ner_dev = os.path.join(data_root, "szeged_ner", "dev")
    morph_basedir = os.path.join(data_root, "morph")
    models = [
        'SZTAKI-HLT/hubert-base-cc',
        "xlm-mlm-100-1280",
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
        "distilbert-base-multilingual-cased",
    ]
    for model in models:
        logging.info("=====================================")
        logging.info(f"=== Morphology {model} ===")
        logging.info("=====================================")
        for task_path in os.scandir(morph_basedir):
            for subword in ['first', 'last']:
                train_file = f"{task_path.path}/train.tsv"
                dev_file = f"{task_path.path}/dev.tsv"
                config = Config.from_yaml(config_fn)
                config.model = 'SentenceRepresentationProber'
                config.dataset_class = 'SentenceProberDataset'
                config.subword_pooling = subword
                config.layer_pooling = 'weighted_sum'
                config.model_name = model
                config.train_file = train_file
                config.dev_file = dev_file
                yield config
                gc.collect()
                torch.cuda.empty_cache()
        logging.info("=====================================")
        logging.info(f"=== Szeged POS {model} ===")
        logging.info("=====================================")
        for layer in [0, -1]:
            for subword in ['first', 'last']:
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
        logging.info("=====================================")
        logging.info(f"=== Full POS {model} ===")
        logging.info("=====================================")
        for layer in [0, -1]:
            for subword in ['first', 'last']:
                config = Config.from_yaml(config_fn)
                config.model = 'TransformerForSequenceTagging'
                config.dataset_class = 'SequenceClassificationWithSubwords'
                config.train_size = 10000
                config.dev_size = 2000
                config.subword_pooling = subword
                config.layer_pooling = layer
                config.model_name = model
                config.train_file = full_pos_train
                config.dev_file = full_pos_dev
                yield config
                gc.collect()
        logging.info("=====================================")
        logging.info(f"=== NER {model} ===")
        logging.info("=====================================")
        for layer in [0, -1]:
            for subword in ['first', 'last']:
                config = Config.from_yaml(config_fn)
                config.model = 'TransformerForSequenceTagging'
                config.dataset_class = 'SequenceClassificationWithSubwords'
                config.subword_pooling = subword
                config.layer_pooling = layer
                config.model_name = model
                config.train_file = szeged_ner_train
                config.dev_file = szeged_ner_dev
                yield config
                gc.collect()
                torch.cuda.empty_cache()

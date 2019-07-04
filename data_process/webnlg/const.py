#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2018/1/24
import logging
import os

logger = logging.getLogger('mylogger')

class Const:
    # triple_len == 3 meas triple is <entity1_end_position, entity_2_end_position, relation>
    triple_len = 3
    home = ''
    origin_train_folder = os.path.join(home, 'train')
    origin_dev_folder = os.path.join(home, 'dev')
    origin_all_train_filename = os.path.join(home, 'origin_all_train.xml')
    origin_all_dev_filename = os.path.join(home, 'origin_all_dev.xml')
    origin_tmp_filename = os.path.join(home, 'tmp.xml')
    origin_example_filename = os.path.join(home, 'origin_example.xml')

    if triple_len == 3:
        folder = 'entity_end_position'

    relations2id_filename = os.path.join(home, folder, 'relations2id.json')
    relation2count_filename = os.path.join(home, folder, 'relation2count.json')
    words2id_filename = os.path.join(home, folder, 'words2id.json')
    words_id2vector_filename = os.path.join(home, folder, 'words_id2vector.json')
    relations_words_id_filename = os.path.join(home, folder, 'relations_words_id.json')
    train_filename = os.path.join(home, folder, 'train.json')
    valid_filename = os.path.join(home, folder, 'valid.json')
    dev_filename = os.path.join(home, folder, 'dev.json')  # this is used as test file
    example_filename = os.path.join(home, folder, 'example.json')
    nyt_style_raw_train_filename = os.path.join(home, folder, 'raw_nyt_style_train.json')
    nyt_style_raw_test_filename = os.path.join(home, folder, 'raw_nyt_style_test.json')
    nyt_style_raw_valid_filename = os.path.join(home, folder, 'raw_nyt_style_valid.json')


if __name__ == '__main__':
    pass
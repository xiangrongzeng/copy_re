#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/12/9
import os


class Const:
    home = ''
    origin_file_path = os.path.join(home, 'origin/')
    origin_train_filename = os.path.join(origin_file_path, 'train.json')
    origin_test_filename = os.path.join(origin_file_path, 'test.json')
    origin_valid_filename = os.path.join(origin_file_path, 'valid.json')
    origin_train_filtered_filename = os.path.join(origin_file_path, 'train_filtered.json')
    raw_train_filename = os.path.join(origin_file_path, 'raw_train.json')
    raw_test_filename = os.path.join(origin_file_path, 'raw_test.json')
    raw_valid_filename = os.path.join(origin_file_path, 'raw_valid.json')
    words2id_filename = os.path.join(home, 'words2id.json')
    relations2id_filename = os.path.join(home, 'relations2id.json')
    relation2count_filename = os.path.join(home, 'relation2count.json')
    vector_bin_filename = os.path.join(home, 'nyt_vec.bin')
    words_id2vector_filename = os.path.join(home, 'words_id2vector.json')
    MAX_SENTENCE_LENGTH = 100

    def __init__(self):
        self.project_name = None

    def seq2seq_re(self):
        self.project_name = os.path.join(self.home, 'seq2seq_re')
        self.raw_test_normal_triple_filename = os.path.join(self.project_name, 'raw_test_normal_triple.json')
        self.raw_test_multi_label_filename = os.path.join(self.project_name, 'raw_test_multi_label.json')
        self.raw_test_overlapping_filename = os.path.join(self.project_name, 'raw_test_overlapping.json')


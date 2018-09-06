#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/8/20
import json
import logging
import os

logger = logging.getLogger('mylogger')


class DataSet():
    NYT = 'nyt'
    CONLL04 = 'conll04'
    WEBNLG = 'webnlg'
    name = None

    @staticmethod
    def set_dataset(dataset_name):
        if dataset_name == DataSet.NYT:
            DataSet.name = DataSet.NYT
        elif dataset_name == DataSet.WEBNLG:
            DataSet.name = DataSet.WEBNLG
        else:
            print 'Dataset %s is not exist!!!!!!!!!! ' % dataset_name
            exit()


class TrainMethod:
    NLL_METHOD = 'NLL'


class DecoderMethod:
    ONE_DECODER = 'ONE'
    MULTI_DECODER = 'MULTI'

    @staticmethod
    def set(idx):
        return [DecoderMethod.ONE_DECODER, DecoderMethod.MULTI_DECODER][idx]


class Config:
    def __init__(self, config_filename=None, cell_name='lstm'):
        home = '/home/sunder/'
        if config_filename is not None:
            print'config filename: %s' % config_filename
            cfg = json.load(open(config_filename, 'r'))
            self.decoder_method = DecoderMethod.set(cfg["decoder_method"])
            self.train_method = TrainMethod.NLL_METHOD
            self.triple_number = cfg["triple_number"]
            self.epoch_number = cfg["epoch_number"]
            self.save_freq = cfg["save_freq"]
            self.encoder_num_units = cfg["encoder_num_units"]
            self.decoder_num_units = cfg["decoder_num_units"]
            self.cell_name = cell_name
            self.learning_rate = cfg["learning_rate"]
            self.batch_size = cfg["batch_size"]
            self.decoder_output_max_length = self.triple_number * 3
            self.dataset_name = cfg["dataset"].lower()
            self.exp_name = cfg["exp_name"]
            DataSet.set_dataset(self.dataset_name)
            model_home = os.path.join(home, 'data/seq2seq_re', DataSet.name, self.exp_name)
            runner = '%s-%s-%s-%s-%s-%s-%s-%s' % (self.dataset_name, self.decoder_method, self.triple_number,
                                                  self.learning_rate, self.batch_size,
                                                  self.cell_name, self.encoder_num_units, self.decoder_num_units)
            self.runner_path = os.path.join(model_home, runner)
        else:
            print 'Config file must be provided.'
            raise

        data_home = os.path.join(home, 'data', DataSet.name)
        if DataSet.name == DataSet.NYT:
            self.words_number = 90760
            self.embedding_dim = 100
            self.relation_number = 25
            self.max_sentence_length = 100
            self.origin_file_path = os.path.join(data_home, 'origin/')
            self.words2id_filename = os.path.join(data_home, 'words2id.json')
            self.relations2id_filename = os.path.join(data_home, 'relations2id.json')
            self.words_id2vector_filename = os.path.join(data_home, 'words_id2vector.json')
            self.raw_train_filename = os.path.join(data_home, 'origin/raw_train.json')
            self.raw_test_filename = os.path.join(data_home, 'origin/raw_test.json')
            self.raw_valid_filename = os.path.join(data_home, 'origin/raw_valid.json')
            self.train_filename = os.path.join(data_home, 'seq2seq_re/train.json')
            self.test_filename = os.path.join(data_home, 'seq2seq_re/test.json')
            self.valid_filename = os.path.join(data_home, 'seq2seq_re/valid.json')
            self.summary_filename = os.path.join(self.runner_path, 'seq2seq_re_graph')
        if DataSet.name == DataSet.WEBNLG:
            self.words_number = 5928
            self.embedding_dim = 100
            self.relation_number = 247
            self.max_sentence_length = 80
            data_home = os.path.join(data_home, 'entity_end_position')
            self.words2id_filename = os.path.join(data_home, 'words2id.json')
            self.relations2id_filename = os.path.join(data_home, 'relations2id.json')
            self.words_id2vector_filename = os.path.join(data_home, 'words_id2vector.json')
            self.train_filename = os.path.join(data_home, 'train.json')
            self.test_filename = os.path.join(data_home, 'dev.json')
            self.valid_filename = os.path.join(data_home, 'valid.json')
            self.summary_filename = os.path.join(self.runner_path, 'seq2seq_re_graph')

        self.NA_TRIPLE = (self.relation_number, self.max_sentence_length, self.max_sentence_length)

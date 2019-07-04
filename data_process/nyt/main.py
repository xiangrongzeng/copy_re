#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/12/9
import json
import logging

import nltk

import utils
from const import Const

logger = logging.getLogger('mylogger')


def run_filter():
    data = utils.read_json(Const.origin_train_filename)
    saved_data = utils.filter_out(data)
    f = open(Const.origin_train_filtered_filename, 'w')
    utils.write_data(f, saved_data)
    print 'filter finish, saved in %s' % Const.origin_train_filtered_filename


def run_split():
    data = utils.read_json(Const.origin_train_filtered_filename)
    print 'splitting'
    test_data, train_data, valid_data = utils.split(data)
    print 'saving'
    utils.write_data(open(Const.raw_test_filename, 'w'), test_data)
    utils.write_data(open(Const.raw_train_filename, 'w'), train_data)
    utils.write_data(open(Const.raw_valid_filename, 'w'), valid_data)


def run_static():
    data = utils.read_json(Const.origin_train_filtered_filename)
    words = utils.static_words(data)
    words2id = dict()
    for i, w in enumerate(words):
        words2id[w] = i
    json.dump(words2id, open(Const.words2id_filename, 'w'), indent=True)

    relations = utils.static_relations(data)
    relations2id = dict()
    for i, r in enumerate(relations):
        relations2id[r] = i
    json.dump(relations2id, open(Const.relations2id_filename, 'w'), indent=True)


def run_word_vectors():
    print 'reading nyt_vec.bin'
    all_w2vec = utils.read_vec_bin()
    words2id = utils.load_words()
    print 'prepare w2vec'
    w2vec = utils.word_vectors(words2id, all_w2vec)
    print 'dumping'
    json.dump(w2vec, open(Const.words_id2vector_filename, 'w'))


def split_triple_type():
    const = Const()
    const.novel_tagging()
    data = utils.read_json(Const.raw_test_filename)
    normal_triple_data = []
    multi_label_data = []
    over_lapping_data = []
    for i, a_data in enumerate(data):
        triples = a_data['relationMentions']
        triples_ = set()
        for triple in triples:
            m1 = nltk.word_tokenize(triple['em1Text'])[-1]
            m2 = nltk.word_tokenize(triple['em2Text'])[-1]
            label = triple['label']
            if label != 'None':
                triples_.add((m1, m2, label))
        triples = []
        for t in triples_:
            triples.extend(list(t))
        if utils.is_normal_triple(triples, is_relation_first=False): normal_triple_data.append(a_data)
        if utils.is_multi_label(triples, is_relation_first=False): multi_label_data.append(a_data)
        if utils.is_over_lapping(triples, is_relation_first=False): over_lapping_data.append(a_data)

    print 'Number of normal triple data %s' % len(normal_triple_data)
    print 'Number of multi triple data %s' % len(multi_label_data)
    print 'Number of overlapping triple data %s' % len(over_lapping_data)
    utils.write_data(open(const.raw_test_normal_triple_filename, 'w'), normal_triple_data)
    utils.write_data(open(const.raw_test_multi_label_filename, 'w'), multi_label_data)
    utils.write_data(open(const.raw_test_overlapping_filename, 'w'), over_lapping_data)


def split_triple_number():
    const = Const()
    const.novel_tagging()
    data = utils.read_json(Const.raw_test_filename)
    # sentences contains 1, 2, 3, 4, and >5 triples
    triples_size_1_data, triples_size_2_data, triples_size_3_data, triples_size_4_data, triples_size_5_data = [], [], [], [], []
    for i, a_data in enumerate(data):
        triples = set()
        for triple in a_data['relationMentions']:
            m1 = nltk.word_tokenize(triple['em1Text'])[-1]
            m2 = nltk.word_tokenize(triple['em2Text'])[-1]
            label = triple['label']
            if label != 'None':
                triples.add((m1, m2, label))

        if len(triples) == 1:
            triples_size_1_data.append(a_data)
        elif len(triples) == 2:
            triples_size_2_data.append(a_data)
        elif len(triples) == 3:
            triples_size_3_data.append(a_data)
        elif len(triples) == 4:
            triples_size_4_data.append(a_data)
        else:
            triples_size_5_data.append(a_data)
    utils.write_data(open(const.raw_test_1_triple_filename, 'w'), triples_size_1_data)
    utils.write_data(open(const.raw_test_2_triple_filename, 'w'), triples_size_2_data)
    utils.write_data(open(const.raw_test_3_triple_filename, 'w'), triples_size_3_data)
    utils.write_data(open(const.raw_test_4_triple_filename, 'w'), triples_size_4_data)
    utils.write_data(open(const.raw_test_5_triple_filename, 'w'), triples_size_5_data)
    print 'Sentence-1-Triple: %d' % len(triples_size_1_data)
    print 'Sentence-2-Triple: %d' % len(triples_size_2_data)
    print 'Sentence-3-Triple: %d' % len(triples_size_3_data)
    print 'Sentence-4-Triple: %d' % len(triples_size_4_data)
    print 'Sentence-5-Triple: %d' % len(triples_size_5_data)


def run_static_relation_freq():
    data = utils.read_json(Const.origin_train_filtered_filename)
    utils.static_relation_freq(data)


if __name__ == '__main__':
    # run_filter()
    # run_split()
    # run_static()
    # run_word_vectors()
    # split_triple_type()
    # split_triple_number()
    run_static_relation_freq()

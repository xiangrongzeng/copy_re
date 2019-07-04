#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2018/1/10
import json
import random

import nltk
import numpy as np

from const import Const


def is_normal_triple(triples, is_relation_first=False):
    """
    normal triples means triples are not over lap in entity.
    example [e1,e2,r1, e3,e4,r2]
    :param triples
    :param is_relation_first
    :return:

    >>> is_normal_triple([1,2,3, 4,5,0])
    True
    >>> is_normal_triple([1,2,3, 4,5,3])
    True
    >>> is_normal_triple([1,2,3, 2,5,0])
    False
    >>> is_normal_triple([1,2,3, 1,2,0])
    False
    >>> is_normal_triple([1,2,3, 4,5,0], is_relation_first=True)
    True
    >>> is_normal_triple([1,2,3, 4,5,3], is_relation_first=True)
    False
    >>> is_normal_triple([1,2,3, 2,5,0], is_relation_first=True)
    True
    >>> is_normal_triple([1,2,3, 1,2,0], is_relation_first=True)
    False
    """
    entities = set()
    for i, e in enumerate(triples):
        key = 0 if is_relation_first else 2
        if i % 3 != key:
            entities.add(e)
    return len(entities) == 2*len(triples)/3


def is_multi_label(triples, is_relation_first=False):
    """
    :param triples:
    :param is_relation_first:
    :return:
    >>> is_multi_label([1,2,3, 4,5,0])
    False
    >>> is_multi_label([1,2,3, 4,5,3])
    False
    >>> is_multi_label([1,2,3, 2,5,0])
    False
    >>> is_multi_label([1,2,3, 1,2,0])
    True
    >>> is_multi_label([1,2,3, 4,5,0], is_relation_first=True)
    False
    >>> is_multi_label([1,2,3, 4,5,3], is_relation_first=True)
    False
    >>> is_multi_label([1,5,0, 2,5,0], is_relation_first=True)
    True
    >>> is_multi_label([1,2,3, 1,2,0], is_relation_first=True)
    False
    """
    if is_normal_triple(triples, is_relation_first):
        return False
    if is_relation_first:
        entity_pair = [tuple(triples[3*i+1: 3*i+3]) for i in range(len(triples)/3)]
    else:
        entity_pair = [tuple(triples[3*i: 3*i+2]) for i in range(len(triples)/3)]
    # if is multi label, then, at least one entity pair appeared more than once
    return len(entity_pair) != len(set(entity_pair))


def is_over_lapping(triples, is_relation_first=False):
    """
    :param triples:
    :param is_relation_first:
    :return:
    >>> is_over_lapping([1,2,3, 4,5,0])
    False
    >>> is_over_lapping([1,2,3, 4,5,3])
    False
    >>> is_over_lapping([1,2,3, 2,5,0])
    True
    >>> is_over_lapping([1,2,3, 1,2,0])
    False
    >>> is_over_lapping([1,2,3, 4,5,0], is_relation_first=True)
    False
    >>> is_over_lapping([1,2,3, 4,5,3], is_relation_first=True)
    True
    >>> is_over_lapping([1,5,0, 2,5,0], is_relation_first=True)
    False
    >>> is_over_lapping([1,2,3, 1,2,0], is_relation_first=True)
    True
    """
    if is_normal_triple(triples, is_relation_first):
        return False
    if is_relation_first:
        entity_pair = [tuple(triples[3*i+1: 3*i+3]) for i in range(len(triples)/3)]
    else:
        entity_pair = [tuple(triples[3*i: 3*i+2]) for i in range(len(triples)/3)]
    # remove the same entity_pair, then, if one entity appear more than once, it's overlapping
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.extend(pair)
    entities = set(entities)
    return len(entities) != 2*len(entity_pair)


def read_vec_bin():
    all_w2vec = dict()
    f = open(Const.vector_bin_filename, 'r')
    for line in f:
        segs = line.strip().split(' ')
        word = segs[0]
        vector = [float(x) for x in segs[1:]]
        all_w2vec[word] = vector
    print 'size %d' % len(all_w2vec)
    return all_w2vec


def word_vectors(words2id, all_w2vec):
    dim = len(all_w2vec[','])
    w2vec = dict()
    for w, idx in words2id.items():
        w2vec[idx] = all_w2vec.get(w, list(np.random.uniform(0, 1, dim)))
    assert len(w2vec) == len(words2id)
    return w2vec


def load_relations():
    return json.load(open(Const.relations2id_filename, 'r'))


def static_relations(data):
    relations = set()
    for a_data in data:
        triples = a_data['relationMentions']
        for triple in triples:
            relation = triple['label']
            relations.add(relation)
    if 'None' in relations:
        relations.remove('None')
    relations = list(relations)
    relations.insert(0, 'None')
    json.dump(relations, open(Const.relations2id_filename, 'w'))
    print 'relation number %d' % len(relations)
    return list(relations)


def static_relation_freq(data):
    relation2count = dict()
    for a_data in data:
        triples = a_data['relationMentions']
        for triple in triples:
            r = triple['label']
            count = relation2count.get(r, 0)
            relation2count[r] = count + 1
    json.dump(relation2count, open(Const.relation2count_filename, 'w'))
    return relation2count


def static_words(data):
    words = set()
    for a_data in data:
        sent_text = a_data['sentText']
        sent_words = nltk.word_tokenize(sent_text)
        words.update(set(sent_words))
    words = list(words)
    words.insert(0, 'UNK')
    print 'words number %d' % len(words)
    return list(words)


def load_words():
    return json.load(open(Const.words2id_filename, 'r'))


def split(data):
    test_instance_num = 5000
    idx = random.sample(range(len(data)), test_instance_num)
    assert len(idx) == test_instance_num
    idx = set(idx)
    test_data = []
    train_data = []
    for i, a_data in enumerate(data):
        if i in idx:
            test_data.append(a_data)
        else:
            train_data.append(a_data)

    valid_instance_num = 5000
    valid_data = train_data[:valid_instance_num]
    train_data = train_data[valid_instance_num:]
    assert len(valid_data) == valid_instance_num
    assert len(test_data) == test_instance_num
    assert len(test_data) + len(train_data) + len(valid_data) == len(data)
    return test_data, train_data, valid_data


#   flag is used to determine if save a sentence if it has no triples
def filter_out(data, flag=False):
    saved_data = []
    for i, a_data in enumerate(data):
        try:
            sent_text = a_data['sentText']
            sent_words = nltk.word_tokenize(sent_text)
            triples_ = a_data['relationMentions']
            triples = set()
            for triple in triples_:
                if triple['label'] != 'None':
                    triples.add((triple['em1Text'], triple['em2Text'], triple['label']))
            if len(sent_words) <= Const.MAX_SENTENCE_LENGTH:
                if len(triples) > 0:
                    saved_data.append(a_data)
                elif flag:
                    saved_data.append(a_data)
        except Exception:
            print a_data['sentText']
        if (i + 1) * 1.0 % 10000 == 0:
            print 'finish %f, %d/%d' % ((i + 1.0) / len(data), (i + 1), len(data))

    print 'instance number %d/%d' % (len(saved_data), len(data))
    return saved_data


def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            a_data = json.loads(line)
            data.append(a_data)
    return data


def write_data(f, data):
    out_data = [json.dumps(d) for d in data]
    for d in out_data:
        f.write(d)
        f.write('\n')
    f.close()



if __name__ == '__main__':
    pass
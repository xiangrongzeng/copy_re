#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/12/12

import json
import logging

import numpy as np

import data_prepare

logger = logging.getLogger('mylogger')


def compare(predict, gold, config, show_rate=None, simple=True):
    normal_triples_gold = []  # normal triples
    normal_triples_predict = []  # normal triples
    multi_label_gold = []  # multi label triples
    multi_label_predict = []  # multi label triples
    over_lapping_gold = []  # overlapping triples
    over_lapping_predict = []  # overlapping triples
    is_relation_first = True
    for p, g in zip(predict, gold):
        if data_prepare.is_normal_triple(g, is_relation_first):
            normal_triples_gold.append(g)
            normal_triples_predict.append(p)
        if data_prepare.is_multi_label(g, is_relation_first):
            multi_label_gold.append(g)
            multi_label_predict.append(p)
        if data_prepare.is_over_lapping(g, is_relation_first):
            over_lapping_gold.append(g)
            over_lapping_predict.append(p)
    f1, precision, recall = compare_(predict, gold, 'ALL', config, show_rate)
    if simple:
        return f1, precision, recall
    compare_(normal_triples_predict, normal_triples_gold, 'Normal-Triples', config, show_rate)
    compare_(multi_label_predict, multi_label_gold, 'Multi-Label', config, show_rate)
    compare_(over_lapping_predict, over_lapping_gold, 'Over-Lapping', config, show_rate)

    # sentences contains 1, 2, 3, 4, and >5 triples
    triples_size_1_gold, triples_size_2_gold, triples_size_3_gold, triples_size_4_gold, triples_size_5_gold = [], [], [], [], []
    triples_size_1_predict, triples_size_2_predict, triples_size_3_predict, triples_size_4_predict, triples_size_5_predict = [], [], [], [], []
    for p, g in zip(predict, gold):
        g_triples = set([tuple(g[i:i + 3]) for i in range(0, len(g), 3)])
        if len(g_triples) == 1:
            triples_size_1_predict.append(p)
            triples_size_1_gold.append(g)
        elif len(g_triples) == 2:
            triples_size_2_predict.append(p)
            triples_size_2_gold.append(g)
        elif len(g_triples) == 3:
            triples_size_3_predict.append(p)
            triples_size_3_gold.append(g)
        elif len(g_triples) == 4:
            triples_size_4_predict.append(p)
            triples_size_4_gold.append(g)
        else:
            triples_size_5_predict.append(p)
            triples_size_5_gold.append(g)
    compare_(triples_size_1_predict, triples_size_1_gold, 'Sentence-1-Triple', config, show_rate)
    compare_(triples_size_2_predict, triples_size_2_gold, 'Sentence-2-Triple', config, show_rate)
    compare_(triples_size_3_predict, triples_size_3_gold, 'Sentence-3-Triple', config, show_rate)
    compare_(triples_size_4_predict, triples_size_4_gold, 'Sentence-4-Triple', config, show_rate)
    compare_(triples_size_5_predict, triples_size_5_gold, 'Sentence-5-Triple', config, show_rate)
    return None, None, None


def _triplelist2triples_(triple_list, config):
    """
     >>> _triplelist2triples_([1,2,3, 2,5,0])
     {(1,2,3),(2,5,0)}
     >>> _triplelist2triples_([1,2,3, 1,2,3, 2,5,0])
     {(1,2,3),(2,5,0)}
     >>> _triplelist2triples_([1,2,3, 2,5,0].extend(config.NA_TRIPLE))
     {(1,2,3),(2,5,0)}
    """
    triple_list = list(triple_list)
    triples = set([tuple(triple_list[i:i + 3]) for i in range(0, len(triple_list), 3)])
    if config.NA_TRIPLE in triples:
        triples.remove(config.NA_TRIPLE)
    return triples


def triples2entities(triples):
    """
    :param triples:
    :return:
    >>> triples2entities([[1,2,3], [0, 3,4]])
    [2,3,4]
    >>> triples2entities([[1,2,3], [1,2,3]])
    [2,3]
    """
    entities = []
    for triple in triples:
        entities.extend(triple[1:])
    return list(set(entities))


def triples2relations(triples):
    """
    :param triples:
    :return:
    >>> triples2relations([[1,2,3], [0, 3,4]])
    [1,0]
    >>> triples2relations([[1,2,3], [1,2,3]])
    [1]
    """
    relations = []
    for triple in triples:
        relations.append(triple[0])
    return list(set(relations))


def error_analyse(predicts, gold, config, entity_or_relation='entity'):
    predict_number = 0
    gold_number = 0
    correct_num = 0
    func = triples2entities if entity_or_relation == 'entity' else triples2relations
    for p, g in zip(predicts, gold):
        p_triples = _triplelist2triples_(p, config)
        g_triples = _triplelist2triples_(g, config)
        p_elements = func(p_triples)
        g_elements = func(g_triples)
        predict_number += len(p_elements)
        gold_number += len(g_elements)
        result = [1 if e in g_elements else 0 for e in p_elements]
        correct_num += sum(result)

    logger.debug('Error Analyse: %s: Predict number %d, Gold number %d, Correct number %d' % (
    entity_or_relation, predict_number, gold_number, correct_num))
    precision = correct_num * 1.0 / predict_number if predict_number > 0 else 0.
    recall = correct_num * 1.0 / gold_number if gold_number > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision * recall > 0 else 0.
    logger.info('Error Analyse: %s: Precision %s, Recall %s, F1 %s' % (entity_or_relation, precision, recall, f1))


def compare_(predict, gold, name, config, show_rate=None, is_show=True):
    predict_number = 0
    gold_number = 0
    correct_num = 0
    for p, g in zip(predict, gold):
        p_triples = _triplelist2triples_(p, config)
        g_triples = _triplelist2triples_(g, config)
        predict_number += len(p_triples)
        gold_number += len(g_triples)

        result = [1 if p_t in g_triples else 0 for p_t in p_triples]
        correct_num += sum(result)

        if np.random.uniform() < show_rate:
            logger.debug('%s: predict %s' % (name, p))
            logger.debug('%s: gold    %s' % (name, g))
            logger.debug(
                '%s: ----------------------------------------------------------------------------- result %s/%s' % (
                name, sum(result), len(g_triples)))

    precision = correct_num * 1.0 / predict_number if predict_number > 0 else 0.
    recall = correct_num * 1.0 / gold_number if gold_number > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision * recall > 0 else 0.
    if is_show:
        logger.info('%s: Instance number %d' % (name, len(gold)))
        logger.info(
            '%s: Predict number %d, Gold number %d, Correct number %d' % (
            name, predict_number, gold_number, correct_num))
        logger.info('%s: Precision %.3f, Recall %.3f, F1 %.3f' % (name, precision, recall, f1))
    return f1, precision, recall


def sent_id2sent_str(sent_id, id2words):
    words = []
    for idx in sent_id:
        if idx > 0:
            try:
                word = id2words[idx]
            except:
                word = 'None-%d' % idx
            words.append(word)
    sent_str = ' '.join(words).encode('utf-8').strip()
    return sent_str


def triple_id2triple_str(triple_id, sent_id, id2words, id2relations, is_relation_first, config):
    assert len(triple_id) == 3
    entity_1_str, entity_2_str, relation_str = 'None', 'None', 'None'
    if is_relation_first:
        r_id, e_1_position_id, e_2_position_id = triple_id[0], triple_id[1], triple_id[2]
    else:
        r_id, e_1_position_id, e_2_position_id = triple_id[2], triple_id[0], triple_id[1]
    if e_1_position_id < config.max_sentence_length:
        try:
            entity_1_str = id2words[sent_id[e_1_position_id]]
        except:
            entity_1_str = 'None-%s-%s' % (e_1_position_id, sent_id[e_1_position_id])
    if e_2_position_id < config.max_sentence_length:
        try:
            entity_2_str = id2words[sent_id[e_2_position_id]]
        except:
            entity_2_str = 'None-%s-%s' % (e_1_position_id, sent_id[e_1_position_id])
    if r_id < config.relation_number:
        try:
            relation_str = id2relations[r_id]
        except:
            relation_str = 'None-%s' % r_id
    return '[%s, %s, %s]' % (
    entity_1_str.encode('utf-8').strip(), entity_2_str.encode('utf-8').strip(), relation_str.encode('utf-8').strip())


def triples2triples_str(triples, sent_id, id2words, id2relations, is_relation_first, config):
    triples_str = []
    for triple in triples:
        triple_string = triple_id2triple_str(triple, sent_id, id2words, id2relations, is_relation_first, config)
        triples_str.append(triple_string)
    return '\t'.join(triples_str)


def _reverse_dict_(a_dict):
    new_dict = {v: k for k, v in a_dict.items()}
    return new_dict


def visualize(sents_id, gold, predict, files_name, config, is_relation_first=True):
    print 'Visualizing ...'
    print config.words2id_filename
    print config.relations2id_filename
    words2id = json.load(open(config.words2id_filename, 'r'))
    relations2id = json.load(open(config.relations2id_filename, 'r'))
    id2words = _reverse_dict_(words2id)
    id2relations = _reverse_dict_(relations2id)

    f1 = open(files_name[0], 'w')
    f2 = open(files_name[1], 'w')
    f3 = open(files_name[2], 'w')
    for d, g, p in zip(sents_id, gold, predict):
        if data_prepare.is_normal_triple(g, is_relation_first):
            f = f1
        elif data_prepare.is_multi_label(g, is_relation_first):
            f = f2
        else:
            f = f3

        f.write(sent_id2sent_str(d, id2words))
        f.write('\n')
        g_triples = _triplelist2triples_(g, config)
        p_triples = _triplelist2triples_(p, config)
        g_triples_string = triples2triples_str(g_triples, d, id2words, id2relations, is_relation_first, config)
        p_triples_string = triples2triples_str(p_triples, d, id2words, id2relations, is_relation_first, config)
        f.write('Gold:   \t' + g_triples_string)
        f.write('\n')
        f.write('Predict:\t' + p_triples_string)
        f.write('\n\n')
    f1.close()
    f2.close()
    f3.close()


if __name__ == '__main__':
    import doctest

    doctest.testmod()

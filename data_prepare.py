#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/8/21
import collections
import json
import logging
import sys
import unicodedata

import nltk
import numpy as np

logger = logging.getLogger('mylogger')

InputData = collections.namedtuple('InputData', ['input_sentence_length',
                                                 'sentence_fw',
                                                 'sentence_bw',
                                                 'sentence_pos_fw',
                                                 'sentence_pos_bw',
                                                 'standard_outputs',
                                                 'input_sentence_append_eos',
                                                 'relations_append_eos',
                                                 'all_triples'])


class Data:
    def __init__(self, data, batch_size, config):
        standard_outputs, sentence_length, sentence_fw, sentence_bw, sentence_pos_fw, sentence_pos_bw, input_sentence_append_eos, relations_append_eos, all_triples_id = data
        self.standard_outputs = np.asanyarray(standard_outputs)
        self.all_triples_id = np.asanyarray(all_triples_id)  # gold triples without padding
        self.sentence_length = np.asanyarray(sentence_length)
        self.sentence_fw = np.asanyarray(sentence_fw)
        self.sentence_bw = np.asanyarray(sentence_bw)
        self.sentence_pos_fw = np.asanyarray(sentence_pos_fw)
        self.sentence_pos_bw = np.asanyarray(sentence_pos_bw)
        self.input_sentence_append_eos = np.asanyarray(input_sentence_append_eos)
        self.relations_append_eos = np.asanyarray(relations_append_eos)
        self.instance_number = len(self.sentence_length)
        self.batch_size = batch_size
        self.batch_index = 0
        self.batch_number = int(self.instance_number / self.batch_size)
        self.config = config

    def next_batch(self, is_random=True):
        if is_random:
            indexes = self.next_random_indexes()
        else:
            indexes = self.next_sequence_indexes()
        all_triples = self.all_triples_id[indexes]
        standard_outputs = self.standard_outputs[indexes]
        batch_data = InputData(input_sentence_length=self.sentence_length[indexes],
                               sentence_fw=self.sentence_fw[indexes],
                               sentence_bw=self.sentence_bw[indexes],
                               sentence_pos_fw=self.sentence_pos_fw[indexes],
                               sentence_pos_bw=self.sentence_pos_bw[indexes],
                               standard_outputs=standard_outputs,
                               input_sentence_append_eos=self.input_sentence_append_eos[indexes],
                               relations_append_eos=self.relations_append_eos[indexes],
                               all_triples=all_triples)
        return batch_data

    #   select data in sequence, mainly for test
    def next_sequence_indexes(self):
        if self.batch_index < self.batch_number:
            indexes = np.asanyarray(range(self.batch_size * self.batch_index, (self.batch_index + 1) * self.batch_size))
            self.batch_index += 1
            return indexes
        else:
            return None

    def reset(self):
        self.batch_index = 0

    # randomly select a batch of data, only for train
    def next_random_indexes(self):
        return np.random.choice(range(self.instance_number), self.batch_size)


def inverse(sent_index):
    inversed = []
    for sent in sent_index:
        sent = list(sent)
        sent.reverse()
        inversed.append(sent)
    return inversed


def padding_sentence(sent_index, config):
    return [padding_a_sentence(sent, config.max_sentence_length) for sent in sent_index]


def padding_a_sentence(sent, max_length):
    sent = list(sent)
    if len(sent) >= max_length:
        return sent[0: max_length]
    for i in range(max_length - len(sent)):
        sent.append(0)
    return sent


def append_eos2sentence(sent_index, config):
    eos_idx = config.words_number
    appended = []
    for sent in sent_index:
        sent = list(sent)
        sent.append(eos_idx)
        appended.append(sent)
    return appended


def padding_triples(all_triples_id, config):
    all_triples_id = [padding_a_triples(triples, config) for triples in all_triples_id]
    return all_triples_id


def padding_a_triples(triples, config):
    """
    Pad triples to given length
    If the given triples is over length, then, randomly select some of it's triples
    :param triples:
    :return: padded triples
    """
    triple_list = triples[:]
    max_length = config.decoder_output_max_length
    triples = [[triple_list[3 * i], triple_list[3 * i + 1], triple_list[3 * i + 2]] for i in
               range(len(triple_list) / 3)]
    np.random.shuffle(triples)
    padded = []
    for t in triples:
        padded.extend(t)

    if len(triple_list) >= max_length:
        padded = padded[: max_length]
    else:
        pad_triple = list(config.NA_TRIPLE)
        for _ in range((max_length - len(triple_list)) / 3):
            padded.extend(pad_triple)
    assert len(padded) == max_length
    return padded


def append_eos2relations(sent_number, config):
    relations_with_eos = range(config.words_number, config.words_number + config.relation_number + 1)
    return [relations_with_eos] * sent_number


def change2relation_first(triples):
    """
    original triple is (entity1, entity2, relation), now, change it as (relation, entity1, entity2)
    :param triples:
    :return: triples with relation first

    >>> change2relation_first([[1, 2, 23, 32, 19, 8],[0,28, 3]])
    [[23, 1, 2, 8, 32, 19], [3, 0, 28]]
    """
    triple_count = 0
    new_triples = []
    for t in triples:
        new = []
        for i in range(len(t) / 3):
            new_t = [t[3 * i + 2], t[3 * i], t[3 * i + 1]]
            new.extend(new_t)
        new_triples.append(new)
        triple_count += len(t) / 3
    logger.info('Gold triple number %d' % triple_count)
    return new_triples


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
    return len(entities) == 2 * len(triples) / 3


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
        entity_pair = [tuple(triples[3 * i + 1: 3 * i + 3]) for i in range(len(triples) / 3)]
    else:
        entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(len(triples) / 3)]
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
        entity_pair = [tuple(triples[3 * i + 1: 3 * i + 3]) for i in range(len(triples) / 3)]
    else:
        entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(len(triples) / 3)]
    # remove the same entity_pair, then, if one entity appear more than once, it's overlapping
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.extend(pair)
    entities = set(entities)
    return len(entities) != 2 * len(entity_pair)


class Prepare:
    def __init__(self, config):
        self.config = config

    def load_words(self):
        return json.load(open(self.config.words2id_filename, 'r'))

    def load_relations(self):
        return json.load(open(self.config.relations2id_filename, 'r'))

    @staticmethod
    def remove_tone(s):
        s = unicodedata.normalize('NFD', s)
        cmb_chrs = dict.fromkeys(c for c in range(sys.maxunicode) if unicodedata.combining(unichr(c)))
        return s.translate(cmb_chrs)

    def load_data(self, name):
        if name.lower() == 'train':
            filename = self.config.train_filename
        elif name.lower() == 'test':
            filename = self.config.test_filename
        elif name.lower() == 'valid':
            filename = self.config.valid_filename
        else:
            print 'name must be "train" or "test", but is %s' % name
            raise ValueError
        print 'loading %s' % filename
        data = json.load(open(filename, 'r'))
        print 'data size %d' % (len(data[0]))
        return data


class NYTPrepare(Prepare):
    @staticmethod
    def read_json(filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                a_data = json.loads(line)
                data.append(a_data)
        return data

    #   flag is used to determine if save a sentence if it has no triples
    def turn2id(self, data, words2id, relations2id, flag=False):
        all_sent_id = []
        all_triples_id = []
        all_sent_length = []
        triples_number = []
        accept_count = 0
        for i, a_data in enumerate(data):
            is_save = True
            sent_text = a_data['sentText']
            sent_id = []
            for w in nltk.word_tokenize(sent_text):
                try:
                    w_id = words2id[w]
                    sent_id.append(w_id)
                except:
                    is_save = False
                    print '[%s] is not in words2id' % w
            triples = a_data['relationMentions']
            triples_id = set()
            for triple in triples:
                # m1 = '_'.join(nltk.word_tokenize(triple['em1Text']))
                # m2 = '_'.join(nltk.word_tokenize(triple['em2Text']))
                m1 = nltk.word_tokenize(triple['em1Text'])[-1]
                m2 = nltk.word_tokenize(triple['em2Text'])[-1]
                label = triple['label']
                if label != 'None':
                    if m2 not in words2id:
                        m2 = self.remove_tone(m2)
                    if m1 not in words2id:
                        m1 = self.remove_tone(m1)
                    try:
                        t_id = (sent_id.index(words2id[m1]), sent_id.index(words2id[m2]),
                                relations2id[label])
                        triples_id.add(t_id)
                    except:
                        is_save = False
                        print '[%s] or [%s] is not in words2id, relation is (%s)' % (m1, m2, label)
            if len(sent_id) <= self.config.max_sentence_length and is_save:
                if flag and len(triples_id) == 0:  # this sentence has no triple and assign a  to it
                    triples_id.add(self.config.NA_TRIPLE)
                    assert len(triples_id) == 1
                if len(triples_id) > 0:
                    accept_count += 1
                    triples = []
                    for t in triples_id:
                        triples.extend(list(t))
                    triples_number.append(len(triples_id))
                    all_triples_id.append(triples)
                    all_sent_id.append(sent_id)
                    all_sent_length.append(len(sent_id))
            if (i + 1) * 1.0 % 1000 == 0:
                print 'finish %f, %d/%d, accept %d' % ((i + 1.0) / len(data), (i + 1), len(data), accept_count)

        assert len(all_triples_id) == len(all_sent_id)
        assert len(all_sent_length) == len(all_sent_id)
        print 'instance number %d/%d' % (len(all_sent_id), len(data))
        print 'triples number max %d, min %d, ave %f' % (
        max(triples_number), min(triples_number), np.mean(triples_number))

        return [all_sent_length, all_sent_id, all_triples_id]

    def prepare(self):
        train_data = self.read_json(self.config.raw_train_filename)
        test_data = self.read_json(self.config.raw_test_filename)
        valid_data = self.read_json(self.config.raw_valid_filename)

        words2id = self.load_words()
        relations2id = self.load_relations()

        print 'processing train data'
        train_data = self.turn2id(train_data, words2id, relations2id)
        json.dump(train_data, open(self.config.train_filename, 'w'))

        print 'processing test data'
        test_data = self.turn2id(test_data, words2id, relations2id)
        json.dump(test_data, open(self.config.test_filename, 'w'))

        print 'processing valid data'
        valid_data = self.turn2id(valid_data, words2id, relations2id)
        json.dump(valid_data, open(self.config.valid_filename, 'w'))
        print 'success'

    #   Above functions are processing raw data
    #   Below functions are prepare the feeding data
    def process(self, data):
        all_sent_length, all_sent_id, all_triples_id = data
        all_triples_id = change2relation_first(all_triples_id)
        standard_outputs = padding_triples(all_triples_id, self.config)
        sentence_length = all_sent_length
        sentence_fw = padding_sentence(all_sent_id, self.config)
        sentence_bw = padding_sentence(inverse(all_sent_id), self.config)
        input_sentence_append_eos = append_eos2sentence(sentence_fw, self.config)
        relations_append_eos = append_eos2relations(len(sentence_fw), self.config)
        return [standard_outputs, sentence_length, sentence_fw, sentence_bw, [None] * len(sentence_fw),
                [None] * len(sentence_fw), input_sentence_append_eos, relations_append_eos, all_triples_id]

    def analyse_data(self, name):
        [_, _, all_triples_id] = self.load_data(name)
        normal_count = 0
        multi_label_count = 0
        over_lapping_count = 0
        for sent_triples in all_triples_id:
            normal_count += 1 if is_normal_triple(sent_triples) else 0
            multi_label_count += 1 if is_multi_label(sent_triples) else 0
            over_lapping_count += 1 if is_over_lapping(sent_triples) else 0
            # if is_normal_triple(sent_triples):
            #     print sent_triples
        print 'Normal Count %d, Multi label Count %d, Overlapping Count %d' % (
        normal_count, multi_label_count, over_lapping_count)
        print 'Normal Rate %f, Multi label Rate %f, Overlapping Rate %f' % \
              (normal_count * 1.0 / len(all_triples_id), multi_label_count * 1.0 / len(all_triples_id),
               over_lapping_count * 1.0 / len(all_triples_id))

        triples_size_1, triples_size_2, triples_size_3, triples_size_4, triples_size_5 = 0, 0, 0, 0, 0
        count_le_5 = 0
        for sent_triples in all_triples_id:
            triples = set([tuple(sent_triples[i:i + 3]) for i in range(0, len(sent_triples), 3)])
            if len(triples) == 1:
                triples_size_1 += 1
            elif len(triples) == 2:
                triples_size_2 += 1
            elif len(triples) == 3:
                triples_size_3 += 1
            elif len(triples) == 4:
                triples_size_4 += 1
            else:
                triples_size_5 += 1
            if len(triples) <= 5:
                count_le_5 += 1
        print 'Sentence number with 1, 2, 3, 4, >5 triplets: %d, %d, %d, %d, %d' % (triples_size_1, triples_size_2,
                                                                                    triples_size_3, triples_size_4,
                                                                                    triples_size_5)
        print 'Sentence number with <= 5 triplets: %d' % count_le_5


class WebNLGPrepare(Prepare):
    def process(self, data):
        all_sent_id, all_triples_id = data
        all_triples_id = change2relation_first(all_triples_id)
        standard_outputs = padding_triples(all_triples_id, self.config)
        sentence_length = [len(sent_id) for sent_id in all_sent_id]
        sentence_fw = padding_sentence(all_sent_id, self.config)
        sentence_bw = padding_sentence(inverse(all_sent_id), self.config)
        input_sentence_append_eos = append_eos2sentence(sentence_fw, self.config)
        relations_append_eos = append_eos2relations(len(sentence_fw), self.config)
        return [standard_outputs, sentence_length, sentence_fw, sentence_bw, [None] * len(sentence_fw),
                [None] * len(sentence_fw), input_sentence_append_eos, relations_append_eos, all_triples_id]


if __name__ == '__main__':
    pass

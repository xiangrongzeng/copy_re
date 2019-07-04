#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2018/1/24
import json
import logging
import os

import utils
from const import Const

logger = logging.getLogger('mylogger')


def generate_origin_file_(folder_name, all_contents):
    file_names = os.listdir(folder_name)
    for file_name in file_names:
        filename = os.path.join(folder_name, file_name)
        if os.path.isdir(filename):
            generate_origin_file_(filename, all_contents)
        else:
            content = utils.direct_read_html(filename)
            all_contents.append(content)


def generate_origin_file():
    for name, out_name in zip([Const.origin_dev_folder, Const.origin_train_folder],
                              [Const.origin_all_dev_filename, Const.origin_all_train_filename]):
        contents = []
        generate_origin_file_(name, contents)
        print len(contents)
        content = utils.combine_htmls(contents)
        print len(content)
        utils.write_list2file(content, out_name)


def statics(name):
    if name == 'train':
        filename = Const.origin_all_train_filename
    elif name == 'dev':
        filename = Const.origin_all_dev_filename
    f = open(filename, 'r')
    content = f.readlines()
    html_doc = ' '.join(content)
    sentences_string, triples_string = utils.parse(html_doc)
    sentences_words = utils.sentence_tokenize(sentences_string)
    relations_words = utils.static_relations(triples_string)
    sentences_words.extend(relations_words)  # not only static the words in sentences, but also the words in relations
    words2id = utils.static_words(sentences_words)
    relations_words_id = [None]
    for r_words in relations_words:
        r_words_id = [utils.turn_word2id(w, words2id) for w in r_words]
        relations_words_id.append(r_words_id)
    json.dump(relations_words_id, open(Const.relations_words_id_filename, 'w'), indent=False)


def run_static_relation_freq(name):
    if name == 'train':
        filename = Const.origin_all_train_filename
    elif name == 'dev':
        filename = Const.origin_all_dev_filename
    f = open(filename, 'r')
    content = f.readlines()
    html_doc = ' '.join(content)
    sentences_string, triples_string = utils.parse(html_doc)
    relation2count = utils.static_relation_freq(triples_string)
    json.dump(relation2count, open(Const.relation2count_filename, 'w'), indent=False)


def prepare(name):
    print name
    if name == 'train':
        filename = Const.origin_all_train_filename
    if name == 'dev':
        filename = Const.origin_all_dev_filename
    if name == 'example':
        filename = Const.origin_example_filename

    print Const.triple_len
    f = open(filename, 'r')
    print filename
    content = f.readlines()
    html_doc = ' '.join(content)
    sentences_string, triples_string = utils.parse(html_doc)
    sentences_words = utils.sentence_tokenize(sentences_string)
    position_triples = utils.find_entity_position(sentences_words, triples_string)
    sentences_word_id, sentence_triples_id = utils.turn2id(sentences_words, position_triples)
    if name == 'train':
        #  split train file into train and valid set
        [valid_sentences_word_id, valid_sentence_triples_id], [train_sentences_word_id, train_sentence_triples_id] = utils.split(sentences_word_id, sentence_triples_id)
        utils.static_triples_info(train_sentence_triples_id)
        utils.triples_type(train_sentence_triples_id)
        utils.static_triples_info(valid_sentence_triples_id)
        utils.triples_type(valid_sentence_triples_id)
        json.dump([train_sentences_word_id, train_sentence_triples_id], open(Const.train_filename, 'w'))
        json.dump([valid_sentences_word_id, valid_sentence_triples_id], open(Const.valid_filename, 'w'))
        utils.instances2nyt_style([train_sentences_word_id, train_sentence_triples_id], Const.nyt_style_raw_train_filename)
        utils.instances2nyt_style([valid_sentences_word_id, valid_sentence_triples_id], Const.nyt_style_raw_valid_filename)
    elif name == 'dev':
        utils.triples_type(sentence_triples_id)
        json.dump([sentences_word_id, sentence_triples_id], open(Const.dev_filename, 'w'))
        utils.instances2nyt_style([sentences_word_id, sentence_triples_id], Const.nyt_style_raw_test_filename)
    else:
        utils.triples_type(sentence_triples_id)
        json.dump([sentences_word_id, sentence_triples_id], open(Const.example_filename, 'w'))

def run_word_vectors():
    print 'reading nyt_vec.bin'
    all_w2vec = utils.read_vec_bin()
    words2id = utils.load_words2id()
    print 'prepare w2vec'
    w2vec = utils.word_vectors(words2id, all_w2vec)
    print 'dumping'
    json.dump(w2vec, open(Const.words_id2vector_filename, 'w'))


def show_instances(class_name=''):
    name = 'train'
    if name == 'train':
        filename = Const.origin_all_train_filename
    elif name == 'dev':
        filename = Const.origin_all_dev_filename
    f = open(filename, 'r')
    content = f.readlines()
    html_doc = ' '.join(content)
    sentences_string, triples_string = utils.parse(html_doc)
    sentences_words = utils.sentence_tokenize(sentences_string)
    position_triples = utils.find_entity_position(sentences_words, triples_string)
    sentences_word_id, sentence_triples_id = utils.turn2id(sentences_words, position_triples)
    utils.triples_type(sentence_triples_id)
    if class_name == 'normal':
        func = utils.is_normal_triple
    elif class_name == 'single_entity_overlap':
        func = utils.is_over_lapping
    else:
        func = utils.is_multi_label

    words2id = utils.load_words2id()
    id2words = {v: k for k, v in words2id.items()}
    for sent_words_id, triples_id in zip(sentences_word_id, sentence_triples_id):
        if func(triples_id, is_relation_first=False):
            print ' '.join([id2words[x] for x in sent_words_id])
            print triples_id
            print '-----------------------------------'


if __name__ == '__main__':
    # generate_origin_file()
    # statics('train')
    # prepare('train')
    # prepare('dev')
    prepare('example')
    # run_word_vectors()
    # show_instances('entity_pair_overlap')
    # run_static_relation_freq('train')

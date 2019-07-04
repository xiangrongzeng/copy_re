#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2018/1/24
import json
import logging
import sys
import unicodedata

import nltk
import numpy as np
from bs4 import BeautifulSoup

from const import Const

logger = logging.getLogger('mylogger')


def combine_htmls(htmls):
    """
    :param htmls:
    :return:
     >>> combine_htmls([['h_1', 'h_2', 'entry_1', 't_2', 't_1'],['h_1', 'h_2', 'entry_2', 'entry_x','t_2', 't_1']])
     ['h_1', 'h_2', 'entry_1', 'entry_2', 'entry_x', 't_2', 't_1']
    """
    head_1 = htmls[0][0]
    head_2 = htmls[0][1]
    tail_2 = htmls[0][-2]
    tail_1 = htmls[0][-1]
    content = [head_1, head_2]
    for html in htmls:
        content.extend(html[2:-2])
    content.append(tail_2)
    content.append(tail_1)
    return content


#  read html line by line, but not parser it
def direct_read_html(filename):
    print 'reading %s' % filename
    content = []
    with open(filename, 'r') as f:
        for line in f:
            content.append(line)
    return content


def write_list2file(content, filename):
    with open(filename, 'w') as f:
        for line in content:
            f.write(line)
    print 'save to %s' % filename


def parse_triple(entry):
    triples = []
    mtriples = entry.modifiedtripleset.children
    for c in mtriples:
        line = c.string.strip()
        if line == '':
            continue
        triple = line.split(' | ')
        try:
            assert len(triple) == 3
            #   <e1, r, e2>  ---->  <e1, e2, r>
            relation = triple[1]
            triple[1] = triple[2]
            triple[2] = relation
            triples.append(triple)
        except:
            print c.string
            print triple
            print "============="
    return triples


def parse_sentence(entry):
    sentences = []
    for lex in entry.find_all('lex'):
        sentences.append(lex.string)
    return sentences


def parse(html_doc):
    print 'parsing html doc'
    triples_string = []
    sentences_string = []
    soup = BeautifulSoup(html_doc, 'html5lib')
    entries = soup.find_all('entry')
    print 'entry number %d' % len(entries)
    for entry in entries:
        triples_string.append(parse_triple(entry))
        sentences_string.append(parse_sentence(entry))
    try:
        assert len(triples_string) == len(entries)
        assert len(sentences_string) == len(entries)
    except:
        print 'triples_string number %d' % len(triples_string)
        print 'sentences_string number %d' % len(sentences_string)
    return sentences_string, triples_string


def static_relations(triples_string):
    relations = set()
    for triples in triples_string:
        for t in triples:
            relations.add(t[2])

    relations = list(relations)
    relation_words = [split_into_lower_tokens(x) for x in relations]
    relations.insert(0, u'None')
    relations2id = dict()
    for idx, r in enumerate(relations):
        relations2id[r] = idx
    print 'relation number %d' % len(relations2id)
    json.dump(relations2id, open(Const.relations2id_filename, 'w'), indent=True)
    return relation_words


def static_relation_freq(triples_string):
    relation2count = dict()
    for triples in triples_string:
        for t in triples:
            r = t[2]
            count = relation2count.get(r, 0)
            relation2count[r] = count + 1

    return relation2count


def load_relations2id():
    return json.load(open(Const.relations2id_filename, 'r'))


def remove_tone(s):
    s = unicodedata.normalize('NFD', s)
    cmb_chrs = dict.fromkeys(c for c in range(sys.maxunicode) if unicodedata.combining(unichr(c)))
    return s.translate(cmb_chrs)


def static_words(sentences_words):
    words = set()
    for sentences in sentences_words:
        for sent_words in sentences:
            words.update(set(sent_words))

    words = list(words)
    words.insert(0, u'PAD')
    words.insert(1, u'UNK')
    words2id = dict()
    for idx, w in enumerate(words):
        words2id[w] = idx
    print 'words number %d' % len(words2id)
    json.dump(words2id, open(Const.words2id_filename, 'w'), indent=True)
    return words2id


def load_words2id():
    return json.load(open(Const.words2id_filename, 'r'))


def sentence_tokenize(sentences_string):
    sentences_words = []
    for sentences in sentences_string:
        sentence_words = []
        for sent in sentences:
            sentence_words.append(nltk.word_tokenize(sent))
        sentences_words.append(sentence_words)
    return sentences_words


def find_entity_position(sentences_words, triples_string):
    valid_count = 0
    multi_words_triple_count = 0  # the triple contains entity that has multi words
    triple_count = 0
    entity_len_static = []
    position_triples = []
    for instance_sentences, instance_triples in zip(sentences_words, triples_string):
        sentence_words = instance_sentences[0]
        instance_position_triples = []
        for triple in instance_triples:
            if Const.triple_len == 3:
                t = _build_entity_last_word_position_triple(triple, sentence_words)
            if Const.triple_len == 5:
                t = _build_entity_start_end_position_triple(triple, sentence_words)
            if t:
                instance_position_triples.extend(t)
                triple_count += 1
                if Const.triple_len == 5:
                    entity_len_static.append(t[1] + 1)
                    entity_len_static.append(t[3] + 1)
                    if t[1] > 0 or t[3] > 0:
                        multi_words_triple_count += 1

        if len(instance_position_triples) != 0:
            valid_count += 1
        position_triples.append(instance_position_triples)
    print 'valid instance number %d' % valid_count
    if Const.triple_len == 5:
        print 'multi words triple number %d, triples number %d' % (multi_words_triple_count, triple_count)
        print 'entity length. AVE: %f, MAX: %d' % (np.mean(entity_len_static), np.max(entity_len_static))
    return position_triples


def _build_entity_last_word_position_triple(triple, sentence_words):
    # used in acl2018
    e1 = triple[0].split('_')[-1]
    e2 = triple[1].split('_')[-1]
    relation_string = triple[2]
    try:
        position1 = sentence_words.index(e1)
        position2 = sentence_words.index(e2)
        return [position1, position2, relation_string]
    except ValueError:
        return None


def _build_entity_start_end_position_triple(triple, sentence_words):
    e1 = triple[0].split('_')
    e2 = triple[1].split('_')
    relation_string = triple[2]
    entity_1_position = _find_entity_start_end_position_(e1, sentence_words)
    entity_2_position = _find_entity_start_end_position_(e2, sentence_words)
    if entity_1_position is None or entity_2_position is None:
        return None
    else:
        entity_1_start_position = entity_1_position[0]
        entity_1_len = entity_1_position[-1] - entity_1_position[0]
        entity_2_start_position = entity_2_position[0]
        entity_2_len = entity_2_position[-1] - entity_2_position[0]
        return [entity_1_start_position, entity_1_len, entity_2_start_position, entity_2_len, relation_string]


# 一个entity的词在句子中的位置
def _find_entity_start_end_position(entity_words, sentence_words):
    best_result = None
    largest_match_len = 0
    for v in range(len(entity_words)):
        start_idx = None
        end_idx = None
        for i, ew in enumerate(entity_words[v:]):
            if ew in sentence_words:
                idx = sentence_words.index(ew)
                if i == 0:
                    start_idx = idx
                    end_idx = start_idx
                else:
                    end_idx = idx
            else:
                break
        if start_idx is not None:
            match_len = end_idx - start_idx + 1
            if match_len > largest_match_len:
                best_result = (start_idx, end_idx)
                largest_match_len = match_len
    return best_result


#  一个entity的词在句子中的位置
def _find_entity_start_end_position_(entity_words, sentence_words):
    i, j = 0, 0
    record = []
    best_shot = []
    while i < len(sentence_words):
        if j >= len(entity_words):
            break
        if sentence_words[i] != entity_words[j]:
            if len(record) > len(best_shot):
                best_shot = record
            j = 0
            record = []
        else:
            record.append(i)
            j += 1
        i += 1
    if len(record) > len(best_shot):
        best_shot = record
    if len(best_shot) == 0:
        best_shot = None
    return best_shot


def turn_word2id(word, word2id):
    if word not in word2id:
        word = 'UNK'
    return word2id[word]


def turn_relation2id(relation, relations2id):
    if relation in relations2id:
        return relations2id[relation]
    else:
        print 'Unknown relation: %s' % relation
        return 0


def turn2id(sentences_words, position_triples):
    sentences_length = []
    words2id = load_words2id()
    relations2id = load_relations2id()
    sentences_word_id = []
    sentence_triples_id = []
    for sentences, triples in zip(sentences_words, position_triples):
        if len(triples) == 0:
            continue
        sentence = sentences[0]
        words_id = [turn_word2id(w, words2id) for w in sentence]
        sentences_word_id.append(words_id)
        assert len(triples) % Const.triple_len == 0
        triples_id = [triples[i] if (i % Const.triple_len) != Const.triple_len - 1
                      else turn_relation2id(triples[i], relations2id) for i in range(len(triples))]
        sentence_triples_id.append(triples_id)
        sentences_length.append(len(sentence))
    print 'average sentence number %.3f, max sentence number %d, min sentence number %d' % (
        np.mean(sentences_length), max(sentences_length), min(sentences_length))
    return sentences_word_id, sentence_triples_id


def static_triples_info(sentences_triples):
    triples_number = []
    max_triple_len = 0
    for triples in sentences_triples:
        assert len(triples) % Const.triple_len == 0
        triples_number.append(len(triples) / Const.triple_len)
        for i, v in enumerate(triples):
            if i % 5 == 1 or i % 5 == 3:
                if max_triple_len < v:
                    max_triple_len = v
    print 'Instance number %d, triples number %d, average triple number %.3f, max triple number %d, min triple number %d' % (
        len(triples_number), sum(triples_number), np.mean(triples_number), max(triples_number), min(triples_number))
    print 'Entity max length %d' % max_triple_len
    print 'Triple distribution: 1: %d, 2: %d, 3: %d, 4: %d, 5: %d, 6: %d, 7: %d' % (
        triples_number.count(1), triples_number.count(2), triples_number.count(3), triples_number.count(4),
        triples_number.count(5), triples_number.count(6), triples_number.count(7))


def split(sentences_words_id, sentences_triples_id):
    valid_number = 500
    assert len(sentences_words_id) == len(sentences_triples_id)
    indexes = range(len(sentences_words_id))
    np.random.shuffle(indexes)
    valid_indexes = indexes[:valid_number]
    valid_sentences_word = []
    valid_sentence_triples = []
    train_sentences_word = []
    train_sentence_triples = []
    for idx, (word_id, triples_id) in enumerate(zip(sentences_words_id, sentences_triples_id)):
        if idx in valid_indexes:
            valid_sentences_word.append(word_id)
            valid_sentence_triples.append(triples_id)
        else:
            train_sentences_word.append(word_id)
            train_sentence_triples.append(triples_id)

    return [valid_sentences_word, valid_sentence_triples], [train_sentences_word, train_sentence_triples]


def is_normal_triple(triples, is_relation_first=False):
    """
    normal triples means triples are not over lap in entity.
    example [[e1,e2,r1], [e3,e4,r2]]
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


def triples_type(sentence_triples_id):
    normal_count = 0
    multi_count = 0
    overlap_count = 0
    for triples in sentence_triples_id:
        if is_normal_triple(triples):
            normal_count += 1
        if is_multi_label(triples):
            multi_count += 1
        if is_over_lapping(triples):
            overlap_count += 1
    print 'Normal number %d, Multi-Label number %d, Overlapping number %d' % (normal_count, multi_count, overlap_count)


def read_vec_bin():
    all_w2vec = dict()
    f = open('/home/sunder/data/nyt/nyt_vec.bin', 'r')
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


def _instance2nyt_style(words, idx, triples, id2relations):
    result = dict()
    result["sentText"] = ' '.join(words).encode('utf-8').strip()
    result["articleId"] = str(idx)
    triples_string = []
    entities = set()
    for i in range(len(triples) / Const.triple_len):
        triple = triples[Const.triple_len * i: Const.triple_len * (i + 1)]
        if Const.triple_len == 3:
            entity1_position = (triple[0], triple[0])
            entity2_position = (triple[1], triple[1])
        if Const.triple_len == 5:
            entity1_position = (triple[0], triple[0] + triple[1])
            entity2_position = (triple[2], triple[2] + triple[3])
        t_string = {"em1Text": entity_position2str(entity1_position, words),
                    "em2Text": entity_position2str(entity2_position, words),
                    "label": id2relations[triples[Const.triple_len * i + 2]].encode('utf-8').strip()}
        triples_string.append(t_string)
        entities.add(t_string['em1Text'])
        entities.add(t_string['em2Text'])
    result["relationMentions"] = triples_string
    entity_mentions = []
    for e in entities:
        e_string = {"start": 0, "label": 'None-label', "text": e}
        entity_mentions.append(e_string)
    result["entityMentions"] = entity_mentions
    return result


def entity_position2str(entity_position, words):
    return ' '.join(words[x].encode('utf-8').strip() for x in range(entity_position[0], entity_position[1]))


def _reverse_dict_(a_dict):
    new_dict = {v: k for k, v in a_dict.items()}
    return new_dict

def instances2nyt_style(instances, filename):
    words_id, triples = instances
    words2id = load_words2id()
    id2words = _reverse_dict_(words2id)
    relations2id = load_relations2id()
    id2relations = _reverse_dict_(relations2id)
    words = [[id2words[idx] for idx in word_id] for word_id in words_id]
    f = open(filename, 'w')
    data = []
    for w, t in zip(words, triples):
        d = _instance2nyt_style(w, 'None', t, id2relations)
        data.append(d)
    write_data(f, data)


def write_data(f, data):
    out_data = [json.dumps(d) for d in data]
    for d in out_data:
        f.write(d)
        f.write('\n')
    f.close()


def split_into_lower_tokens(inp):
    """
    Split the input string into lower case tokens

    Input string can be seem as a sentence, words in this sentence are separate by space.
    However, a word may be camel style (such as 'ApplePal' or 'wifeOf') or underline style (such as 'apple_pal').
    We gonna split words in camel style or underline style into tokens too.

    :param inp: str, sentence with words
    :return: list

    >>> split_into_lower_tokens("ReferenceNumber in the National Register of Historic Places")
    ['reference', 'number', 'in', 'the', 'national', 'register', 'of', 'historic', 'places']

    >>> split_into_lower_tokens('3rd_runway_SurfaceType')
    ['3rd', 'runway', 'surface', 'type']

    >>> split_into_lower_tokens('owner')
    ['owner']

    >>> split_into_lower_tokens('LibraryofCongressClassification')
    ['libraryof', 'congress', 'classification']
    """
    words = split_space(inp)
    tokens = []
    for w in words:
        tokens.extend(split_underline(w))
    result = []
    for w in tokens:
        result.extend(split_camel(w))
    return [x.lower() for x in result]


def split_camel(inp):
    """Split the input string in Camel style into tokens

    :param inp: str, input string
    :return: list, tokens

    >>> split_camel('WifeOf')
    ['Wife', 'Of']

    >>> split_camel('wifeOf')
    ['wife', 'Of']

    >>> split_camel('wifeof')
    ['wifeof']

    >>> split_camel('wife_of')
    ['wife_of']
    """
    tokens = []
    token_element = []
    for x in inp:
        if x.isalpha() and x == x.upper():
            token = ''.join(token_element)
            if token != '':
                tokens.append(token)
            token_element = []
        token_element.append(x)
    tokens.append(''.join(token_element))
    return tokens


def split_underline(inp):
    """
    Split the input string in underline style into tokens

    :param inp: string, underline style
    :return: list, tokens

    >>> split_underline('wife_of')
    ['wife', 'of']

    >>> split_underline('wifeOf')
    ['wifeOf']

    >>> split_underline('wife of')
    ['wife of']
    """
    tokens = inp.split('_')
    return tokens


def split_space(inp):
    """
    Split the input string with space

    :param inp: string
    :return: list, tokens

    >>> split_space('wife_of')
    ['wife_of']

    >>> split_space('wifeOf')
    ['wifeOf']

    >>> split_space('wife of')
    ['wife', 'of']
    """
    tokens = inp.split(' ')
    return tokens


if __name__ == '__main__':
    import doctest

    doctest.testmod()

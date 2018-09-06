#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/8/19
import json
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

logger = logging.getLogger('mylogger')


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def get_embedding_table(config):
    if os.path.isfile(config.words_id2vector_filename):
        logger.info('Word Embedding init from %s' % config.words_id2vector_filename)
        words_id2vec = json.load(open(config.words_id2vector_filename, 'r'))
        words_vectors = [0] * len(words_id2vec)
        for id, vec in words_id2vec.items():
            words_vectors[int(id)] = vec
        words_embedding_table = tf.Variable(name='words_emb_table', initial_value=words_vectors, dtype=tf.float32)
    else:
        logger.info('Word Embedding random init')
        words_embedding_table = tf.get_variable(name='words_emb_table',
                                                shape=[config.words_number + 1, config.embedding_dim],
                                                dtype=tf.float32)
    relation_and_eos_embedding_table = tf.get_variable(name='relation_and_eos_emb_table',
                                                       shape=[config.relation_number + 1, config.embedding_dim],
                                                       dtype=tf.float32)
    embedding_table = tf.concat([words_embedding_table, relation_and_eos_embedding_table], axis=0,
                                name='embedding_table')
    return embedding_table


def get_pos_embedding_table(config):
    return tf.get_variable(name='pos_emb_table', shape=[config.pos_number, config.embedding_dim], dtype=tf.float32)


def set_rnn_cell(name, num_units):
    if name.lower() == 'gru':
        return tf.contrib.rnn.GRUCell(num_units)
    elif name.lower() == 'lstm':
        return tf.contrib.rnn.LSTMCell(num_units)
    else:
        return tf.contrib.rnn.BasicRNNCell(num_units)


class Encoder:
    def __init__(self, config, max_sentence_length, embedding_table):
        self.max_sentence_length = max_sentence_length
        self.encoder_fw_cell = None
        self.encoder_bw_cell = None
        self.embedding_table = embedding_table
        self.pos_embedding_table = embedding_table
        self.input_sentence_fw_pl = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, max_sentence_length],
                                                   name='input_sentence_fw')
        self.input_sentence_pos_fw_pl = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, max_sentence_length],
                                                       name='input_sentence_pos_fw')
        self.input_sentence_length = tf.placeholder(dtype=tf.int32, shape=[config.batch_size],
                                                    name='input_sentence_length')
        self.outputs = None
        self.state = None
        self.config = config

    def set_cell(self, name, num_units):
        with tf.variable_scope('encoder'):
            self.encoder_fw_cell = set_rnn_cell(name, num_units)
            self.encoder_bw_cell = set_rnn_cell(name, num_units)

    def _encode(self, inputs):
        try:
            words_pl, pos_pl = inputs
            words_vector = tf.nn.embedding_lookup(self.embedding_table, words_pl)
            pos_vector = tf.nn.embedding_lookup(self.pos_embedding_table, pos_pl)
            input_vector = tf.concat((words_vector, pos_vector), axis=-1)
        except:
            input_vector = tf.nn.embedding_lookup(self.embedding_table, inputs)
        logger.debug('Input vector shape %s' % input_vector.get_shape())
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_fw_cell,
                                                         cell_bw=self.encoder_bw_cell,
                                                         inputs=input_vector,
                                                         dtype=tf.float32)

        if self.config.cell_name == 'lstm':
            logger.debug('Encoder before concat: output shape %s,%s' % (len(outputs), outputs[0].get_shape()))
            logger.debug('Encoder before concat: state shape %s,%s' % (np.shape(state), state[0][0].get_shape()))
            outputs = tf.concat(outputs, axis=-1)
            state = (
                tf.reduce_mean((state[0][0], state[1][0]), axis=0), tf.reduce_mean((state[0][1], state[1][1]), axis=0))
            logger.debug('Encoder: outputs shape %s' % outputs.get_shape())
            logger.debug('Encoder: state shape %s,%s' % (np.shape(state), state[0].get_shape()))
        elif self.config.cell_name == 'gru':
            outputs, state = tf.reduce_mean(outputs, axis=0), tf.reduce_mean(state, axis=0)
            logger.debug('Encoder: outputs shape %s' % outputs.get_shape())
            logger.debug('Encoder: state shape %s' % state.get_shape())
        else:
            logger.error('Undefined cell name %s' % self.config.cell_name)
            exit()
        return outputs, state

    def build(self, is_use_pos=False):
        logger.info('Encoding')
        with tf.variable_scope('seq_encoder'):
            if is_use_pos:
                inputs = [self.input_sentence_fw_pl, self.input_sentence_pos_fw_pl]
            else:
                inputs = self.input_sentence_fw_pl
            self.outputs, self.state = self._encode(inputs=inputs)


class Decoder:
    def __init__(self, decoder_output_max_length, embedding_table, encoder, config):
        self.config = config
        self.decoder_cell_number = self.config.decoder_output_max_length / 3
        self.embedding_table = embedding_table
        self.decoder_output_max_length = decoder_output_max_length

        self.encoder = encoder
        self.input_sentence_length = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size],
                                                    name='input_sentence_length')
        self.input_sentence_append_eos_pl = tf.placeholder(dtype=tf.int32,
                                                           shape=[self.config.batch_size,
                                                                  self.config.max_sentence_length + 1],
                                                           name='input_sentence_append_eos_pl')
        self.relations_append_eos_pl = tf.placeholder(dtype=tf.int32,
                                                      shape=[self.config.batch_size, self.config.relation_number + 1],
                                                      name='relations_append_eos')
        self.sparse_standard_outputs = tf.placeholder(dtype=tf.int64,
                                                      shape=[self.config.batch_size,
                                                             self.config.decoder_output_max_length],
                                                      name='standard_outputs')
        self.batch_bias4copy = tf.constant(
            value=[i * (self.config.max_sentence_length + 1) for i in range(self.config.batch_size)],
            dtype=tf.int64,
            name='batch_bias4copy')
        self.batch_bias4predict = tf.constant(
            value=[i * (self.config.relation_number + 1) for i in range(self.config.batch_size)],
            dtype=tf.int64,
            name='batch_bias4predict')

        self.decode_cell = None
        self.actions_by_time = None
        self.probs_by_time = []
        self.picked_actions_prob = None
        self.losses = None
        self.opt = None
        self.cell_num_units = None
        self.tmp = []

    def set_cell(self, name, num_units):
        pass

    def do_predict(self, inputs):
        with tf.variable_scope('predict'):
            W = tf.get_variable(name='W',
                                shape=[int(inputs.get_shape()[-1]), self.config.relation_number],
                                dtype=tf.float32)
            b = tf.get_variable(name='b',
                                shape=self.config.relation_number,
                                dtype=tf.float32)
            logits = selu(tf.matmul(inputs, W)) + b
            return logits

    @staticmethod
    def do_eos(inputs):
        with tf.variable_scope('eos'):
            W = tf.get_variable(name='W',
                                shape=[int(inputs.get_shape()[-1]), 1],
                                dtype=tf.float32)
            b = tf.get_variable(name='b',
                                shape=1,
                                dtype=tf.float32)
            logits = selu(tf.matmul(inputs, W)) + b
            return logits

    @staticmethod
    def do_copy(inputs, encoder_states):
        #   encoder_states的shape是[batch_size, max_sentence_length, hidden_dim]，现在转换为一个list，
        #   list中的每个元素的shape是[batch_size, hidden_dim]， list中一共有max_sentence_length个这样的元素
        # encoder_states = self.encoder.outputs
        encoder_states_by_time = tf.unstack(encoder_states, axis=1)
        with tf.variable_scope('copy'):
            W = tf.get_variable(name='W',
                                shape=[int(encoder_states.get_shape()[-1]) + int(inputs.get_shape()[-1]), 1],
                                dtype=tf.float32)
            values = []
            for states in encoder_states_by_time:
                att_value = selu(tf.matmul(tf.concat((states, inputs), axis=1), W))
                values.append(att_value)
            values = tf.stack(values)
            values = tf.squeeze(values, -1)
            values = tf.transpose(values)
        return values

    @staticmethod
    def calc_context(decoder_state, encoder_outputs):
        encoder_states_by_time = tf.unstack(encoder_outputs, axis=1)
        with tf.variable_scope('calc_context'):
            W = tf.get_variable(name='W',
                                shape=[int(encoder_outputs.get_shape()[-1]) + int(decoder_state.get_shape()[-1]), 1],
                                dtype=tf.float32)
            values = []
            for states in encoder_states_by_time:
                att_value = selu(tf.matmul(tf.concat((states, decoder_state), axis=1), W))
                values.append(att_value)
            values = tf.stack(values)
            values = tf.squeeze(values, -1)
            values = tf.nn.softmax(tf.transpose(values))
            att_values = tf.unstack(values, axis=1)
            all = []
            for att_value, state in zip(att_values, encoder_states_by_time):
                att_value = tf.expand_dims(att_value, axis=1)
                all.append(att_value * state)
            context_vector = tf.reduce_mean(tf.stack(all), axis=0)
        logger.debug('context_vector shape %s' % context_vector.get_shape())
        return context_vector

    @staticmethod
    def combine_inputs(states):
        [inputs, context_vector] = states
        with tf.variable_scope('combine_state'):
            W = tf.get_variable(name='W',
                                shape=[sum([int(s.get_shape()[-1]) for s in states]),
                                       int(states[0].get_shape()[-1])],
                                dtype=tf.float32)
            states = tf.concat(states, axis=1)
            tf.get_variable_scope().reuse_variables()
        return tf.matmul(states, W)

    def build(self, is_train=True):
        pass

    def get_prob(self, probs, indexes):
        depth = probs.get_shape()[-1]
        one_hot = tf.one_hot(indexes, depth)
        probs = tf.reduce_sum(probs * one_hot, axis=1)
        return probs

    def _loss(self):
        logging.info('Calculating loss')
        probs = tf.reshape(self.picked_actions_prob, [-1])
        probs = tf.clip_by_value(probs, 1e-10, 1.0)
        self.losses = tf.reduce_mean(-tf.log(probs))

    def _optimize(self):
        logging.info('Optimizing')
        learning_rate = self.config.learning_rate
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.losses)

    def update(self, data, sess):
        feed_dict = {self.input_sentence_length: data.input_sentence_length,
                     self.encoder.input_sentence_fw_pl: data.sentence_fw,
                     self.encoder.input_sentence_length: data.input_sentence_length,
                     self.input_sentence_append_eos_pl: data.input_sentence_append_eos,
                     self.relations_append_eos_pl: data.relations_append_eos,
                     self.sparse_standard_outputs: data.standard_outputs}
        loss_val, _ = sess.run([self.losses, self.opt], feed_dict=feed_dict)
        return loss_val

    def predict(self, data, sess):
        feed_dict = {self.input_sentence_length: data.input_sentence_length,
                     self.sparse_standard_outputs: data.standard_outputs,
                     self.encoder.input_sentence_fw_pl: data.sentence_fw,
                     self.encoder.input_sentence_length: data.input_sentence_length,
                     self.input_sentence_append_eos_pl: data.input_sentence_append_eos,
                     self.relations_append_eos_pl: data.relations_append_eos}
        actions = sess.run(self.actions_by_time, feed_dict=feed_dict)
        return actions


class OneDecoder(Decoder):
    def set_cell(self, name, num_units):
        self.cell_num_units = num_units
        with tf.variable_scope('cell'):
            cell = set_rnn_cell(name, num_units)
        self.decode_cell = cell

    def build(self, is_train=True):
        with tf.variable_scope('seq_decoder'):
            sentence_eos_embedding = tf.nn.embedding_lookup(self.embedding_table, self.input_sentence_append_eos_pl)
            sentence_eos_embedding = tf.reshape(sentence_eos_embedding,
                                                shape=[self.config.batch_size * (self.config.max_sentence_length + 1),
                                                       self.config.embedding_dim])
            relations_eos_embedding = tf.nn.embedding_lookup(self.embedding_table, self.relations_append_eos_pl)
            relations_eos_embedding = tf.reshape(relations_eos_embedding,
                                                 shape=[self.config.batch_size * (self.config.relation_number + 1),
                                                        self.config.embedding_dim])

            #   开始解码的输入GO
            go = tf.get_variable(name='GO', shape=[1, self.config.embedding_dim])

            #   设置mask
            #   no matter when only copy or only predict, end_of_sentence symbol can also be generated
            mask_only_copy = tf.ones(shape=[self.config.batch_size, self.config.max_sentence_length], dtype=tf.float32)
            mask_eos = tf.ones(shape=[self.config.batch_size, 1], dtype=tf.float32)

            #   解码，并保存解码的结果
            actions_by_time = []
            probs_by_time = []
            sparse_standard_outputs_by_time = tf.unstack(self.sparse_standard_outputs, axis=1)

            with tf.variable_scope('rnn'):
                inputs = tf.nn.embedding_lookup(go, tf.zeros(shape=[self.config.batch_size], dtype=tf.int64))
                decode_state = self.encoder.state
                copy_history = tf.zeros(shape=[self.config.batch_size, self.config.max_sentence_length],
                                        dtype=tf.float32)
                logger.debug('Decoder: cell_num_units %s' % str(self.cell_num_units))
                logger.debug('Decoder: state shape %s' % str(np.shape(decode_state)))
                for i in range(self.config.decoder_output_max_length):
                    logger.info('%s Decoding of %2d/%-2d' % (
                        self.decode_cell.name, i + 1, self.config.decoder_output_max_length))
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    if i % 3 == 1 or i % 3 == 0:
                        c_mask = tf.concat([mask_only_copy, mask_eos], axis=1)
                    if i % 3 == 2:
                        c_mask = tf.concat([c_mask, mask_eos], axis=1)

                    if self.config.cell_name == 'gru':
                        decode_state_h = decode_state
                    elif self.config.cell_name == 'lstm':
                        decode_state_h = decode_state[0]
                    logger.debug('decode_state_h shape %s' % decode_state_h.get_shape())
                    context_vector = self.calc_context(decode_state_h, self.encoder.outputs)
                    inputs = self.combine_inputs([inputs, context_vector])
                    logger.debug('Decoder: inputs shape %s' % str(inputs.get_shape()))
                    if self.config.cell_name == 'gru':
                        decode_state = decode_state_h
                        logger.debug('Decoder: state shape %s' % decode_state.get_shape())
                    elif self.config.cell_name == 'lstm':
                        decode_state = (decode_state_h, decode_state[1])
                        logger.debug(
                            'Decoder: state shape %s,%s' % (np.shape(decode_state), decode_state[0].get_shape()))
                    # decode
                    output, decode_state = self.decode_cell(inputs, decode_state)

                    # eos
                    eos_logits = self.do_eos(output)
                    # copy
                    copy_logits_ = self.do_copy(output, self.encoder.outputs)
                    logger.debug('Decoder: copy_logits shape %s' % str(copy_logits_.get_shape()))
                    copy_logits = tf.concat((copy_logits_, eos_logits), axis=1) * c_mask
                    copy = tf.nn.softmax(copy_logits)
                    logger.debug('Decoder: copy shape %s' % str(copy.get_shape()))
                    # predict
                    predict_logits = self.do_predict(output)
                    logger.debug('Decoder: predict_logits shape %s' % str(predict_logits.get_shape()))
                    predict_logits = tf.concat((predict_logits, eos_logits), axis=1)
                    predict = tf.nn.softmax(predict_logits)
                    logger.debug('Decoder: predict shape %s' % str(predict.get_shape()))

                    # select action
                    if i % 3 == 2 or i % 3 == 1:
                        action_logits = copy_logits
                        action_probs = copy
                        copy_history += copy_logits_
                    else:
                        action_logits = predict_logits
                        action_probs = predict
                    max_action = tf.squeeze(tf.argmax(action_logits, 1))

                    picked_actions = max_action
                    actions_by_time.append(picked_actions)

                    probs = self.get_prob(action_probs, sparse_standard_outputs_by_time[i])
                    probs_by_time.append(probs)

                    # look up the embedding of the copied word or selected relation
                    if i % 3 == 2 or i % 3 == 1:
                        inputs = tf.nn.embedding_lookup(sentence_eos_embedding,
                                                        picked_actions + self.batch_bias4copy)
                    else:
                        inputs = tf.nn.embedding_lookup(relations_eos_embedding,
                                                        picked_actions + self.batch_bias4predict)

                    if i % 3 == 1:  # mask the already copied position
                        #  in every triple, one position should be copied at most once.
                        #  use mask to mask the already copied position, but eos should not be masked
                        copy_position_one_hot = tf.cast(tf.one_hot(picked_actions, self.config.max_sentence_length + 1),
                                                        tf.float32)
                        copy_position_one_hot = copy_position_one_hot[:, :-1]  # remove the mask of eos
                        c_mask = mask_only_copy * (1. - copy_position_one_hot)

            self.actions_by_time = tf.stack(actions_by_time, axis=1)
            self.probs_by_time = tf.stack(probs_by_time, axis=1)

            if is_train:
                logging.info('Prepare for loss')
                self.picked_actions_prob = self.probs_by_time
                self._loss()
                self._optimize()


class MultiDecoder(Decoder):
    def set_cell(self, name, num_units):
        self.cell_num_units = num_units
        self.decode_cell = []
        for i in range(self.decoder_cell_number):
            with tf.variable_scope('cell_%d' % i):
                cell = set_rnn_cell(name, num_units)
            self.decode_cell.append(cell)

    def build(self, is_train=True):
        with tf.variable_scope('seq_decoder'):
            sentence_eos_embedding = tf.nn.embedding_lookup(self.embedding_table, self.input_sentence_append_eos_pl)
            sentence_eos_embedding = tf.reshape(sentence_eos_embedding,
                                                shape=[self.config.batch_size * (self.config.max_sentence_length + 1),
                                                       self.config.embedding_dim])
            relations_eos_embedding = tf.nn.embedding_lookup(self.embedding_table, self.relations_append_eos_pl)
            relations_eos_embedding = tf.reshape(relations_eos_embedding,
                                                 shape=[self.config.batch_size * (self.config.relation_number + 1),
                                                        self.config.embedding_dim])

            #   开始解码的输入GO
            go = tf.get_variable(name='GO', shape=[1, self.config.embedding_dim])

            #   设置mask
            #   no matter when only copy or only predict, end_of_sentence symbol can also be generated
            mask_only_copy = tf.ones(shape=[self.config.batch_size, self.config.max_sentence_length], dtype=tf.float32)
            mask_eos = tf.ones(shape=[self.config.batch_size, 1], dtype=tf.float32)

            #   解码，并保存解码的结果
            actions_by_time = []
            probs_by_time = []
            sparse_standard_outputs_by_time = tf.unstack(self.sparse_standard_outputs, axis=1)

            with tf.variable_scope('rnn'):
                logger.debug('Decoder: cell_num_units %s' % str(self.cell_num_units))
                if self.config.cell_name == 'gru':
                    previous_state = tf.zeros(shape=[self.config.batch_size, self.cell_num_units], dtype=tf.float32)
                elif self.config.cell_name == 'lstm':
                    previous_state = (tf.zeros(shape=[self.config.batch_size, self.cell_num_units], dtype=tf.float32),
                                      tf.zeros(shape=[self.config.batch_size, self.cell_num_units], dtype=tf.float32))
                for cell_idx in range(self.decoder_cell_number):
                    inputs = tf.nn.embedding_lookup(go, tf.zeros(shape=[self.config.batch_size], dtype=tf.int64))
                    if self.config.cell_name == 'gru':
                        decode_state = tf.reduce_mean((self.encoder.state, previous_state), axis=0)
                    elif self.config.cell_name == 'lstm':
                        decode_state = (tf.reduce_mean((self.encoder.state[0], previous_state[0]), axis=0),
                                        tf.reduce_mean((self.encoder.state[1], previous_state[1]), axis=0))
                    with tf.variable_scope('decoder_%d' % cell_idx):
                        logger.debug('Decoder: state shape %s' % str(np.shape(decode_state)))
                        for t in range(3):  # predict 3 elements of a triple
                            logger.info('%s Decoding of %d-%d/%d' % (
                                self.decode_cell[cell_idx].name, t + 1, cell_idx + 1, self.decoder_cell_number))
                            if t > 0: tf.get_variable_scope().reuse_variables()
                            if t == 0 or t == 1:
                                c_mask = tf.concat([mask_only_copy, mask_eos], axis=1)
                            else:
                                c_mask = tf.concat([c_mask, mask_eos], axis=1)

                            if self.config.cell_name == 'gru':
                                decode_state_h = decode_state
                            elif self.config.cell_name == 'lstm':
                                decode_state_h = decode_state[0]
                            context_vector = self.calc_context(decode_state_h, self.encoder.outputs)
                            inputs = self.combine_inputs([inputs, context_vector])
                            if self.config.cell_name == 'gru':
                                decode_state = decode_state_h
                                logger.debug('Decoder: state shape %s' % decode_state.get_shape())
                            elif self.config.cell_name == 'lstm':
                                decode_state = (decode_state_h, decode_state[1])
                                logger.debug('Decoder: state shape %s,%s' % (
                                    np.shape(decode_state), decode_state[0].get_shape()))
                            # decode
                            output, decode_state = self.decode_cell[cell_idx](inputs, decode_state)

                            # eos
                            eos_logits = self.do_eos(output)
                            # copy
                            copy_logits = self.do_copy(output, self.encoder.outputs)
                            logger.debug('Decoder: copy_logits shape %s' % str(copy_logits.get_shape()))
                            copy_logits = tf.concat((copy_logits, eos_logits), axis=1) * c_mask
                            copy = tf.nn.softmax(copy_logits)
                            logger.debug('Decoder: copy shape %s' % str(copy.get_shape()))
                            # predict
                            predict_logits = self.do_predict(output)
                            logger.debug('Decoder: predict_logits shape %s' % str(predict_logits.get_shape()))
                            predict_logits = tf.concat((predict_logits, eos_logits), axis=1)
                            predict = tf.nn.softmax(predict_logits)
                            logger.debug('Decoder: predict shape %s' % str(predict.get_shape()))

                            # select action
                            if t == 2 or t == 1:
                                action_logits = copy_logits
                                action_probs = copy
                            else:
                                action_logits = predict_logits
                                action_probs = predict
                            max_action = tf.squeeze(tf.argmax(action_logits, 1))

                            picked_actions = max_action
                            actions_by_time.append(picked_actions)

                            probs = self.get_prob(action_probs, sparse_standard_outputs_by_time[3 * cell_idx + t])
                            probs_by_time.append(probs)

                            # look up the embedding of the copied word or selected relation
                            if t == 2 or t == 1:
                                inputs = tf.nn.embedding_lookup(sentence_eos_embedding,
                                                                picked_actions + self.batch_bias4copy)
                            else:
                                inputs = tf.nn.embedding_lookup(relations_eos_embedding,
                                                                picked_actions + self.batch_bias4predict)

                            if t == 1:  # mask the already copied position
                                #  in every triple, one position should be copied at most once.
                                #  use mask to mask the already copied position, but eos should not be masked
                                copy_position_one_hot = tf.cast(
                                    tf.one_hot(picked_actions, self.config.max_sentence_length + 1),
                                    tf.float32)
                                copy_position_one_hot = copy_position_one_hot[:, :-1]  # remove the mask of eos
                                c_mask = mask_only_copy * (1. - copy_position_one_hot)

                        # previous state is the state of previous decoder
                        previous_state = decode_state
            self.actions_by_time = tf.stack(actions_by_time, axis=1)
            self.probs_by_time = tf.stack(probs_by_time, axis=1)

            if is_train:
                logging.info('Prepare for loss')
                self.picked_actions_prob = self.probs_by_time
                self._loss()
                self._optimize()


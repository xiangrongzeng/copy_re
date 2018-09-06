#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/8/21
import argparse
import json
import logging
import logging.config
import os

import numpy as np
import tensorflow as tf

import const
import data_prepare
import evaluation
import model
from const import DecoderMethod

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='configfile', type=str, help='path of the config file')
parser.add_argument('-t', dest='is_train', type=int, default=0, choices=[0, 1, 2],
                    help='0 for train, 1 for test and 2 for valid')
parser.add_argument('-cell', dest='cell_name', type=str, default='lstm', help='cell name: lstm or gru')
parser.add_argument('-g', dest='gpu', type=str, default='', help='gpu id')

args = parser.parse_args()
config_filename = args.configfile
cell_name = args.cell_name
is_train = [True, False, False][args.is_train]
train_test_valid = ['Train', 'Test', 'Valid'][args.is_train]
# 调用配置
config = const.Config(config_filename=config_filename, cell_name=cell_name)
gpu = args.gpu
#   set the batch size in test. Because the test data size maybe smaller then batch size
if not is_train:
    if config.dataset_name == const.DataSet.NYT:
        config.batch_size = 1000
    if config.dataset_name == const.DataSet.CONLL04 or config.dataset_name == const.DataSet.WEBNLG:
        config.batch_size = 2
logger = logging.getLogger('mylogger')


def setup_logging(default_path='logging.json',
                  default_level=logging.DEBUG,
                  env_key='LOG_CFG', ):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            log_config = json.load(f)
            handlers = log_config['handlers']
            log_folder = os.path.join(config.runner_path, 'logfile')
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            handlers['debug_file_handler']['filename'] = os.path.join(log_folder, 'debug.log')
            handlers['info_file_handler']['filename'] = os.path.join(log_folder, 'info.log')
            handlers['error_file_handler']['filename'] = os.path.join(log_folder, 'error.log')
            log_config['handlers'] = handlers
        logging.config.dictConfig(log_config)
    else:
        logging.basicConfig(level=default_level)


setup_logging()

logger.info('Decoder_method: %s-%s, %s, triple_number: %s, learn_rate %s, batch_size: %s, epoch_num: %s, gpu: %s'
            % (config.decoder_method, config.train_method, train_test_valid,
               config.triple_number, config.learning_rate, config.batch_size,
               config.epoch_number, gpu if gpu else None))
logger.info('runner: %s' % config.runner_path)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True


def test_model(data, decoder, sess, show_rate, is_visualize, simple=True):
    sents_id = []
    predictes = []
    gold = []
    for batch_i in range(data.batch_number):
        batch_data = data.next_batch(is_random=False)
        predict_answer = decoder.predict(batch_data, sess)
        gold_answer = batch_data.all_triples
        predictes.extend(predict_answer)
        gold.extend(gold_answer)
        sents_id.extend(batch_data.sentence_fw)
    try:
        assert len(predictes) == len(gold)
    except AssertionError:
        logger.info('Error, predictes number (%d) not equal gold number (%d)' % (len(predictes), len(gold)))
        exit()
    f1, precision, recall = evaluation.compare(predictes, gold, config, show_rate, simple=simple)
    if not simple:
        evaluation.error_analyse(predictes, gold, config, entity_or_relation='entity')
        evaluation.error_analyse(predictes, gold, config, entity_or_relation='relation')

    if is_visualize:
        visualize_normal_file = os.path.join(config.runner_path, 'visualize_normal_instance.txt')
        visualize_multi_file = os.path.join(config.runner_path, 'visualize_multi_instance.txt')
        visualize_overlap_file = os.path.join(config.runner_path, 'visualize_overlap_instance.txt')
        print visualize_normal_file
        print visualize_multi_file
        print visualize_overlap_file
        evaluation.visualize(sents_id, gold, predictes,
                             [visualize_normal_file, visualize_multi_file, visualize_overlap_file], config)
    return f1, precision, recall


def test_all_models(model_epochs, data, decoder, sess, config):
    if train_test_valid.lower() == 'test':
        filename = os.path.join(config.runner_path, 'test_result.txt')
    elif train_test_valid.lower() == 'valid':
        filename = os.path.join(config.runner_path, 'valid_result.txt')
    else:
        logger.error('Error, illegal instruction: {}'.format(train_test_valid))
        raise
    out_file = open(filename, 'w')
    saver = tf.train.Saver()
    for epoch in model_epochs:
        model_name = 'model-{}'.format(epoch)
        model_filename = os.path.join(config.runner_path, model_name)
        logger.info('Test model: {}'.format(model_name))
        saver.restore(sess, model_filename)
        data.reset()
        f1, precision, recall = test_model(data, decoder, sess, show_rate=None, is_visualize=False)
        out_file.write('%d,%.3f,%.3f,%.3f' % (epoch, precision, recall, f1))
        out_file.write('\n')
        out_file.flush()
    out_file.close()


def train_NLL_model(data, epoch_range, decoder, sess):
    saver = tf.train.Saver(max_to_keep=60)
    for epoch_i in epoch_range:
        epoch_loss = []
        for batch_i in range(data.batch_number):
            batch_data = data.next_batch(is_random=True)
            loss_val = decoder.update(batch_data, sess)
            epoch_loss.append(loss_val)
        logger.info('NLL Train: epoch %-3d, loss %f' % (epoch_i, np.mean(epoch_loss)))

        if config.dataset_name == const.DataSet.NYT:
            remainder = 0
        if config.dataset_name == const.DataSet.WEBNLG:
            remainder = 1
        if epoch_i % config.save_freq == remainder:
            save_path = os.path.join(config.runner_path, 'model')
            saver.save(sess, save_path=save_path, global_step=epoch_i)
            logger.info('Saved model {0}-{1}'.format(save_path, epoch_i))


def get_model(train_method, config):
    logger.info('Building model --------------------------------------')
    logger.info('Parameter init Randomly')
    embedding_table = model.get_embedding_table(config)
    encoder = model.Encoder(config=config, max_sentence_length=config.max_sentence_length,
                            embedding_table=embedding_table)
    encoder.set_cell(name=config.cell_name, num_units=config.encoder_num_units)
    encoder.build()

    if config.decoder_method == DecoderMethod.ONE_DECODER:
        decoder = model.OneDecoder(decoder_output_max_length=config.decoder_output_max_length,
                                   embedding_table=embedding_table,
                                   encoder=encoder, config=config)
    elif config.decoder_method == DecoderMethod.MULTI_DECODER:
        decoder = model.MultiDecoder(decoder_output_max_length=config.decoder_output_max_length,
                                     embedding_table=embedding_table,
                                     encoder=encoder, config=config)
    else:
        logger.error('decoder_method is %s, which is illegal.' % config.decoder_method)
        exit()

    decoder.set_cell(name=config.cell_name, num_units=config.decoder_num_units)
    decoder.build(is_train=is_train)

    sess = tf.Session(config=tfconfig)

    sess.run(tf.global_variables_initializer())
    logger.debug('print trainable variables')
    for v in tf.trainable_variables():
        value = sess.run(v)
        logger.debug('Name %s:\tmean %s, max %s, min %s' % (v.name, np.mean(value), np.max(value), np.min(value)))

    return decoder, sess


if __name__ == '__main__':
    if config.dataset_name == const.DataSet.NYT:
        prepare = data_prepare.NYTPrepare(config)
    elif config.dataset_name == const.DataSet.WEBNLG:
        prepare = data_prepare.WebNLGPrepare(config)
    else:
        print 'illegal dataset name: %s' % config.dataset_name
        exit()

    decoder, sess = get_model(train_method=config.train_method, config=config)
    # decoder, sess = None, None

    logger.info('Prepare {} data'.format(train_test_valid))
    data = prepare.load_data(train_test_valid.lower())
    data = prepare.process(data)
    data = data_prepare.Data(data, config.batch_size, config)

    if is_train:
        logger.info('****************************** NLL Train ******************************')
        train_NLL_model(data, epoch_range=range(1, config.epoch_number + 1), decoder=decoder, sess=sess)
    else:
        logger.info(
            '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {} Dataset $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'.format(train_test_valid))

        # ===================
        # epoch = 1
        # saver = tf.train.Saver()
        # model_name = 'model-{}'.format(epoch)
        # model_filename = os.path.join(config.runner_path, model_name)
        # logger.info('Test model: {}'.format(model_name))
        # saver.restore(sess, model_filename)
        # test_model(data, decoder=decoder, sess=sess, show_rate=None, is_visualize=True, simple=False)

        # =============
        model_epoch = range(1, 51, 1)
        # model_epoch = range(1, 100, 2)
        test_all_models(model_epoch, data, decoder, sess, config)

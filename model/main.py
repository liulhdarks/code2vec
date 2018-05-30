from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'lihua.llh'

import tensorflow as tf
import numpy as np
from model.datas import *
from model.model import *
from model.iterator import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import math

import sys
from os import path

root_dir = path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
sys.path.append(root_dir)
sys.path.append('.')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('random_seed', 123, "random_seed")
flags.DEFINE_float('lr_min', 0.00001, "lr_min")
flags.DEFINE_float('lr_max', 0.003, "lr_max")
flags.DEFINE_float('move_avg_decay', 0.999, "move_avg_decay")
flags.DEFINE_string('optimize_algo', 'adam', "optimize_algo")
flags.DEFINE_float('keep_prob', 0.5, "keep_prob")
flags.DEFINE_integer('max_grad_norm', 4, "max_grad_norm")
flags.DEFINE_integer('max_path_length', 80, "max_path_length")
flags.DEFINE_integer('batch_size', 32, "batch_size")
flags.DEFINE_integer('terminal_embed_size', 100, "terminal_embed_size")
flags.DEFINE_integer('path_embed_size', 100, "path_embed_size")
flags.DEFINE_integer('encode_size', 100, "encode_size")
flags.DEFINE_integer('attention_size', 100, "attention_size")
flags.DEFINE_integer('num_sampled', 32, "num_sampled")
flags.DEFINE_string('model_path', "./model", "model_path")
flags.DEFINE_string('summary_path', "./summary", "summary_path")
flags.DEFINE_string('corpus_path', "../dataset/corpus.txt", "corpus_path")
flags.DEFINE_string('path_idx_path', "../dataset/path_idxs.txt", "path_idx_path")
flags.DEFINE_string('terminal_idx_path', "../dataset/terminal_idxs.txt", "terminal_idx_path")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


class Option(object):
    def __init__(self, reader):
        self.max_path_length = FLAGS.max_path_length
        self.terminal_embed_size = FLAGS.terminal_embed_size
        self.path_embed_size = FLAGS.path_embed_size
        self.encode_size = FLAGS.encode_size
        self.attention_size = FLAGS.attention_size
        self.terminal_count = reader.terminal_idxs.count()
        self.path_count = reader.path_idxs.count()
        self.label_count = reader.label_idx.count()
        self.num_sampled = FLAGS.num_sampled
        self.training = True
        self.keep_prob = FLAGS.keep_prob
        self.test = False


def build_metric():
    with tf.variable_scope('metric'):
        precision_holder = tf.placeholder(dtype=tf.float32, name='precision_holder')
        recall_holder = tf.placeholder(dtype=tf.float32, name='recall_holder')
        f1_holder = tf.placeholder(dtype=tf.float32, name='f1_holder')
        acc_holder = tf.placeholder(dtype=tf.float32, name='accuracy_holder')
        precision_var = tf.Variable(0, dtype=tf.float32, name="precision", trainable=False)
        recall_var = tf.Variable(0, dtype=tf.float32, name="recall", trainable=False)
        f1_var = tf.Variable(0, dtype=tf.float32, name="f1", trainable=False)
        acc_var = tf.Variable(0, dtype=tf.float32, name="accuracy", trainable=False)
        p_update = tf.assign(precision_var, precision_holder)
        r_update = tf.assign(recall_var, recall_holder)
        f1_update = tf.assign(f1_var, f1_holder)
        acc_update = tf.assign(acc_var, acc_holder)
        metric_update = tf.group(p_update, r_update, f1_update, acc_update)
        tf.summary.scalar('precision', precision_var)
        tf.summary.scalar('recall', recall_var)
        tf.summary.scalar('f1', f1_var)
        tf.summary.scalar('accuracy', acc_var)
        return metric_update, {'p': precision_holder, 'r': recall_holder, 'f1': f1_holder, 'acc': acc_holder}


def train():
    reader = DataReader(FLAGS.corpus_path, FLAGS.path_idx_path, FLAGS.terminal_idx_path)
    opt = Option(reader)

    builder = DatasetBuilder(reader, opt)
    train_dataset = builder.train_dataset
    test_dataset = builder.test_dataset
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)
    batch_datas = iterator.get_next()
    inputs_start = batch_datas['inputs_start']
    inputs_path = batch_datas['inputs_path']
    inputs_end = batch_datas['inputs_end']
    labels = batch_datas['labels']

    metric_update, metric = build_metric()

    with tf.variable_scope('model'):
        train_opt = Option(reader)
        train_opt.training = True
        lr = tf.placeholder(dtype=tf.float32, name='lr')
        train_model = Code2vecModel(inputs_start, inputs_path, inputs_end, labels, train_opt)
        train_op, global_step = utils.optimize_loss_batch_norm(train_model.loss, FLAGS.optimize_algo, lr, max_grad_norm=FLAGS.max_grad_norm,
                                                               move_avg_decay=FLAGS.move_avg_decay, momentum=0.9, opt_decay=0.9,
                                                               global_step_val=0)
    with tf.variable_scope('model', reuse=True):
        eval_opt = Option(reader)
        eval_opt.training = False
        eval_model = Code2vecModel(inputs_start, inputs_path, inputs_end, labels, eval_opt)

    with tf.variable_scope('model', reuse=True):
        test_opt = Option(reader)
        test_opt.training = False
        test_opt.test = True
        start_holder = tf.placeholder(dtype=tf.int32, shape=[None, opt.max_path_length], name='start_holder')
        path_holder = tf.placeholder(dtype=tf.int32, shape=[None, opt.max_path_length], name='path_holder')
        end_holder = tf.placeholder(dtype=tf.int32, shape=[None, opt.max_path_length], name='end_holder')
        test_model = Code2vecModel(start_holder, path_holder, end_holder, None, test_opt)

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init_op)

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.summary_path, sess.graph)

        saver = tf.train.Saver()
        # saver.restore(sess, '/Users/lihua.llh/Desktop/studio/oss/force/model/summary/embeddings.bin-55909')

        last_loss = None
        last_accuracy = None
        bad_count = 0
        lr_min = FLAGS.lr_min
        lr_max = FLAGS.lr_max
        lr_decay = 10000
        iter_num = 0
        _lr = lr_max
        lr_iter_num = 0
        for i in range(10000):
            train_loss = 0
            count = 0
            while True:
                iter_num += 1
                lr_iter_num += 1
                try:
                    _lr = lr_min + (lr_max - lr_min) * math.exp(-float(lr_iter_num + 1) / lr_decay)
                    if iter_num > 1 and iter_num % 200 == 0:
                        _, summary_str, loss, step = sess.run([train_op, merged_summary, train_model.loss, global_step], feed_dict={lr: _lr})
                        summary_writer.add_summary(summary_str, step)
                    else:
                        _, loss, step = sess.run([train_op, train_model.loss, global_step], feed_dict={lr: _lr})
                    train_loss += loss
                    count += 1
                except tf.errors.OutOfRangeError:
                    sess.run(train_init_op)
                    break
            test_loss, accuracy, p, r, f1 = eval(sess, eval_model, batch_datas, test_init_op)
            sess.run(metric_update, feed_dict={metric['p']: p, metric['r']: r, metric['f1']: f1, metric['acc']: accuracy})
            sess.run(train_init_op)
            if i % 1 == 0:
                print(i, ' loss:', train_loss, ' test_loss:', test_loss, ' acc:', accuracy, ' p:', p,
                      ' r:', r, ' f1:', f1, ' best:', last_loss,
                      ' bad:', bad_count, ' lr:', _lr, ' step:', step)
            if i > 1 and i % 40 == 0:
                test(reader, sess, eval_model, batch_datas, test_init_op)
                sess.run(train_init_op)
            if last_loss is None or train_loss < last_loss or last_accuracy is None or last_accuracy < accuracy:
                if last_accuracy is None or last_accuracy < accuracy:
                    export_graph(sess)
                last_loss = train_loss
                last_accuracy = accuracy
                bad_count = 0
                output_file = path.join(FLAGS.summary_path, "embeddings.bin")
                saver.save(sess, output_file, global_step=step)
            else:
                bad_count += 1
                if bad_count % 2 == 0 and lr_decay > 1000:
                    lr_decay -= 200
            if bad_count > 10:
                print('early stop loss:', train_loss, ' bad:', bad_count)
                test(reader, sess, eval_model, batch_datas, test_init_op)
                break


def eval(sess, model, batch_datas, test_init_op):
    labels = batch_datas['labels']
    sess.run(test_init_op)
    sum_loss = 0
    cl_labels_idx = []
    cl_preds_idx = []
    while True:
        try:
            loss, labels_idx, outputs = sess.run([model.loss, labels, model.outputs])
            sum_loss += loss
            preds_idx = np.argmax(outputs, axis=1)
            cl_labels_idx.extend(labels_idx)
            cl_preds_idx.extend(preds_idx)
        except tf.errors.OutOfRangeError:
            break
    p = precision_score(cl_labels_idx, cl_preds_idx, average='weighted')
    r = recall_score(cl_labels_idx, cl_preds_idx, average='weighted')
    f1score = f1_score(cl_labels_idx, cl_preds_idx, average='weighted')
    accuracy = accuracy_score(cl_labels_idx, cl_preds_idx)
    return sum_loss, accuracy, p, r, f1score


def test(reader, sess, model, batch_datas, test_init_op):
    inputs_start = batch_datas['inputs_start']
    inputs_path = batch_datas['inputs_path']
    inputs_end = batch_datas['inputs_end']
    labels = batch_datas['labels']
    sess.run(test_init_op)
    start, path, end, tags, outputs, probs = sess.run([inputs_start, inputs_path, inputs_end, labels, model.outputs, model.attn_probs])
    for i in range(len(start)):
        start_datas = [reader.terminal_idxs.i2t[v] for v in start[i]]
        path_datas = [reader.path_idxs.i2t[v] for v in path[i]]
        end_datas = [reader.terminal_idxs.i2t[v] for v in end[i]]
        tag_datas = reader.label_idx.get_label(tags[i])
        pred_idx = np.argmax(outputs[i])
        pred_data = reader.label_idx.get_label(pred_idx)
        prob = probs[i]
        datas = zip(start_datas, path_datas, end_datas, prob[:len(start_datas)])
        if tag_datas == pred_data or i == len(start) - 1:
            for s, p, e, pb in datas:
                print(s, ' ', p, ' ', e, ' p:', pb)
            print('tag_datas:', tag_datas)
            print('pred_data:', pred_data)
            break


def export_graph(sess):
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                    output_node_names=[
                                                                        'model_2/train_classify/outputs',
                                                                        'model_2/attention/attn_probs'])
    for node in output_graph_def.node:
        node.device = ""
    output_file = path.join(FLAGS.model_path, "code2vec_model.pb")
    with tf.gfile.FastGFile(output_file, mode='wb') as f:
        f.write(output_graph_def.SerializeToString())


def main(_):
    train()


if __name__ == '__main__':
    np.random.seed(123)
    tf.set_random_seed(123)
    random.seed(123)
    tf.app.run()

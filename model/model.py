import tensorflow as tf
from model import utils


class Code2vecModel(object):
    def __init__(self, inputs_start, inputs_path, inputs_end, labels, opt):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.terminal_embedding = tf.get_variable('terminal_embedding',
                                                 [opt.terminal_count, opt.terminal_embed_size],
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 dtype=tf.float32)
            self.path_embedding = tf.get_variable('path_embedding', [opt.path_count, opt.path_embed_size],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=tf.float32)
            if opt.training:
                tf.summary.histogram('terminal_embedding', self.terminal_embedding)
                tf.summary.histogram('path_embedding', self.path_embedding)
            inputs = self.build_inputs(inputs_start, inputs_path, inputs_end)
        self.inputs_len = utils.retrieve_seq_length_op3(inputs_start)
        encode_inputs = self.build_encode_inputs(inputs, opt)
        self.attn_probs, attn_outputs = self.build_attention(encode_inputs, opt)
        self.loss, self.outputs = self.build_classify(opt, attn_outputs, labels, opt.num_sampled)
        if opt.training:
            tf.summary.scalar('loss', self.loss)

    def build_inputs(self, inputs_start, inputs_path, inputs_end):
        embed_start = tf.nn.embedding_lookup(self.terminal_embedding, inputs_start)
        embed_path = tf.nn.embedding_lookup(self.path_embedding, inputs_path)
        embed_end = tf.nn.embedding_lookup(self.terminal_embedding, inputs_end)

        inputs = tf.concat([embed_start, embed_path, embed_end], axis=2)
        print('inputs:', inputs)
        return inputs

    def build_encode_inputs(self, inputs, opt):
        with tf.name_scope('encode'):
            inputs_size = inputs.get_shape()[2]
            encode_weights = tf.get_variable('encode_weights',
                                             [inputs_size, opt.encode_size],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=tf.float32)
            # encode_inputs = tf.nn.tanh(utils.multiply_tensors(inputs, encode_weights))
            seq_len = inputs.get_shape()[1]
            inputs = tf.reshape(inputs, [-1, inputs_size])
            encode_inputs = tf.nn.tanh(tf.matmul(inputs, encode_weights))
            if opt.training and opt.keep_prob < 1.0:
                encode_inputs = tf.nn.dropout(encode_inputs, opt.keep_prob)
            encode_inputs = tf.reshape(encode_inputs, [-1, seq_len, opt.encode_size])
            print('encode_inputs:', encode_inputs)
            return encode_inputs

    def build_attention(self, inputs, opt):
        with tf.name_scope('attention'):
            seq_len = inputs.get_shape()[1]
            inputs_size = inputs.get_shape()[2]
            attention_weights = tf.get_variable('attention_weights',
                                             [opt.attention_size, 1],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=tf.float32)
            flat_inputs = tf.reshape(inputs, [-1, inputs_size])
            attn_inputs = tf.matmul(flat_inputs, attention_weights)
            # attn_inputs = tf.reshape(attn_inputs, [-1, seq_len, 1])
            # print('attn_inputs:', attn_inputs)
            # attn_inputs = tf.nn.softmax(attn_inputs, dim=1)
            attn_inputs = tf.reshape(attn_inputs, [-1, seq_len])
            y_mask = tf.sequence_mask(self.inputs_len, maxlen=seq_len, dtype=tf.float32)
            print('y_mask:', y_mask)
            attn_inputs = utils.mask_inf(attn_inputs, y_mask)

            attn_inputs = tf.nn.softmax(attn_inputs, name='attn_probs')
            print('attn_inputs:', attn_inputs)
            attn_inputs = tf.expand_dims(attn_inputs, dim=-1)
            print('attn_inputs expand:', attn_inputs)
            attn_outputs = tf.reduce_sum(tf.multiply(inputs, attn_inputs), axis=1)
            print('attn_outputs:', attn_outputs)
            return attn_inputs, attn_outputs

    def build_classify(self, opt, query_output, label_target, num_sampled=16):
        with tf.name_scope('train_classify'):
            print('query_output:', query_output)
            state_size = query_output.get_shape()[1]
            label_count = opt.label_count
            classify_weight = tf.get_variable('classify_weight', [label_count, state_size],
                                                              initializer=tf.contrib.layers.xavier_initializer(),
                                                              dtype=tf.float32)
            classify_bias = tf.get_variable("classify_bias", [label_count], dtype=tf.float32,
                                                            initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            projection_weights = tf.transpose(classify_weight)
            label_preds = tf.nn.relu(tf.matmul(query_output, projection_weights) + classify_bias)
            outputs = tf.nn.softmax(label_preds, name='outputs')
            print('outputs:', outputs)
            if not opt.test:
                if num_sampled is not None:
                    dims_label_target = tf.expand_dims(label_target, -1)
                    print('query_output:', query_output)
                    print('classify_weight:', classify_weight)
                    print('dims_label_target:', dims_label_target)
                    classify_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                        weights=classify_weight,
                        biases=classify_bias,
                        labels=dims_label_target,
                        inputs=query_output,
                        num_sampled=num_sampled,
                        num_classes=label_count,
                        partition_strategy="div"), name='train_classify_loss')
                else:
                    label_target_onehot = tf.to_float(tf.one_hot(label_target, label_count, 1, 0))
                    classify_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=label_target_onehot, logits=label_preds))
                return classify_loss, outputs
            else:
                return None, outputs



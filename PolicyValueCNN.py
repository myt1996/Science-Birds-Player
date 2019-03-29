import os

import numpy as np
import tensorflow as tf


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
##os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
def conv_layer_3(pre_layer, filters=128, batch_norm=True, trainable=True):
    out = tf.layers.conv2d(inputs=pre_layer, filters=filters, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None, trainable=trainable)
    if batch_norm is True:
        out = tf.layers.batch_normalization(out, axis=3)
    out = tf.nn.relu(out)
    return out

def conv_layer_1(pre_layer, filters=32, batch_norm=True, trainable=True):
    out = tf.layers.conv2d(inputs=pre_layer, filters=filters, kernel_size=[1, 1], strides=(1, 1), padding="same", activation=None, trainable=trainable)
    if batch_norm is True:
        out = tf.layers.batch_normalization(out, axis=3)
    out = tf.nn.relu(out)
    return out

def fc_layer(pre_layer, units, trainable=True):
    return tf.layers.dense(inputs=pre_layer, units=units, activation=tf.nn.relu, trainable=trainable)

def residual_block(pre_layer, filters=128, kernel_size=[3, 3], strides=(1, 1), trainable=True):
    plus1_conv_relu = tf.layers.conv2d(inputs=pre_layer, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, trainable=trainable)
    plus1_conv_relu = tf.layers.batch_normalization(plus1_conv_relu, axis=3)
    plus1_conv_relu = tf.nn.relu(plus1_conv_relu)
    plus2_conv = tf.layers.conv2d(inputs=plus1_conv_relu, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", trainable=trainable)
    plus2_conv = tf.layers.batch_normalization(plus2_conv, axis=3)
    block_output_layer = pre_layer + plus2_conv
    block_output_layer = tf.nn.relu(block_output_layer)
    return block_output_layer

class PolicyValueCNN():
    def __init__(self, state_shape, action_count, model_file=None):

        self.x = state_shape[0]
        self.y = state_shape[1]
        self.channel = state_shape[2] #state shape must include three value for x and y and channel axis
        self.action_count = action_count

        self.states = tf.placeholder(tf.float32, shape=[None, self.y, self.x, self.channel])
        self.mcts_probs = tf.placeholder(tf.float32, shape=[None, self.action_count])
        self.values = tf.placeholder(tf.float32, shape=[None, 1])

        #modify this value if need
        self.pre_conv_layer_count = 1
        self.pre_conv_layer_filter_count = 128

        self.residual_block_count = 0
        
        self.policy_conv_layer_filter_count = 4
        self.policy_fc_layer_count = 0
        self.policy_fc_layer_unit_count = 128

        self.value_conv_layer_filter_count = 2
        self.value_fc_layer_count = 1
        self.value_fc_layer_unit_count = 128

        # pre conv layer is used before blocks, must need

        self.conv1 = conv_layer_3(self.states, 32)
        self.conv2 = conv_layer_3(self.conv1, 64)
        self.conv3 = conv_layer_3(self.conv2, 128)
        self.block = self.conv3

        #self.pre_conv = conv_layer_3(self.states, self.pre_conv_layer_filter_count)
        #for _ in range(1, self.pre_conv_layer_count):
        #    self.pre_conv = conv_layer_3(self.pre_conv, self.pre_conv_layer_filter_count)

        ## blocks is main part of nn but not must need
        #for i in range(self.residual_block_count):
        #    if i == 0:
        #        self.block = residual_block(self.pre_conv, self.pre_conv_layer_filter_count)
        #    else:
        #        self.block = residual_block(self.block, self.pre_conv_layer_filter_count)

        # policy conv fc and out
        self.policy_conv = conv_layer_1(self.block, self.policy_conv_layer_filter_count)
        self.policy_conv_flatten = tf.layers.flatten(self.policy_conv)
        if self.policy_fc_layer_count > 0:
            self.policy_fc = fc_layer(self.policy_conv_flatten, self.policy_fc_layer_unit_count)
            for _ in range(1, self.policy_fc_layer_count):
                self.policy_fc = fc_layer(self.policy_fc, self.policy_fc_layer_unit_count)
            self.policy_out_real = tf.layers.dense(inputs=self.policy_fc, units=self.action_count)
        else:
            self.policy_out_real = tf.layers.dense(inputs=self.policy_conv_flatten, units=self.action_count)
        self.policy_out = tf.nn.log_softmax(self.policy_out_real)
        
        # value conv fc and out
        self.value_conv = conv_layer_1(self.block, self.value_conv_layer_filter_count)
        self.value_conv_flatten = tf.layers.flatten(self.value_conv)
        if self.value_fc_layer_count > 0:
            self.value_fc = fc_layer(self.value_conv_flatten, self.value_fc_layer_unit_count)
            for _ in range(1, self.value_fc_layer_count):
                self.value_fc = fc_layer(self.value_fc, self.value_fc_layer_unit_count)
            self.value_out_real = tf.layers.dense(inputs=self.value_fc, units=1)
        else:
            self.value_out_real = tf.layers.dense(inputs=self.value_conv_flatten, units=1)
        self.value_out = tf.nn.tanh(self.value_out_real)

        # loss
        self.policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.policy_out_real, labels=self.mcts_probs))
        self.value_loss = tf.losses.mean_squared_error(predictions=self.value_out, labels=self.values)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        self.loss = self.policy_loss + self.value_loss + l2_penalty

        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.session = tf.Session()

        self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.exp(self.policy_out) * self.policy_out, 1)))

        init = tf.global_variables_initializer()
        self.session.run(init)
        
        self.saver = tf.train.Saver(max_to_keep=10000)
        if model_file is not None: 
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        log_act_probs, value = self.session.run([self.policy_out, self.value_out],
                feed_dict={self.states: state_batch})
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, result_batch, lr):
        loss, entropy, _ = self.session.run([self.loss, self.entropy, self.optimizer],
                feed_dict={self.states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.values: result_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def test_step(self, state_batch, mcts_probs, result_batch, lr):
        loss = self.session.run(self.loss,
                feed_dict={self.states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.values: result_batch,
                           self.learning_rate: lr})
        return loss

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)

if __name__ == '__main__':
    cnn = PolicyValueCNN([210,120,12], 147)

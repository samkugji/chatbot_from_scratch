import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layer_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import LSTMStateTuple, MultiRNNCell, RNNCell, DropoutWrapper, LayerNormBasicLSTMCell

from configs import model_config


class ChatSeq2SeqModel(object):
    """
    Multi-layer seq2seq with attention network using highway connections
    """

    def __init__(self, config, use_lstm=True, forward_only=False):
        self.vocab_size = model_config.vocab_size
        self.batch_size = model_config.batch_size
        self.enc_hidden_size = model_config.enc_hidden_size
        self.enc_num_layers = model_config.enc_num_layers
        self.dec_hidden_size = model_config.dec_hidden_size


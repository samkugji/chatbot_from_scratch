import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from configs import model_config
from lib import chat_seq2seq_model_old, data_utils


def create_model(session, config, forward_only):
    model = chat_seq2seq_model_old.ChatSeq2SeqModel(
        config=config,
        forward_only=forward_only
    )

    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    # if model files existed, load model file
    if ckpt and gfile.Exists("{}.index".format(ckpt.model_checkpoint_path)):
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    # initialize parameters and create new mode
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model
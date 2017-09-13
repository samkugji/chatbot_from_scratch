import numpy as np
import tensorflow as tf
import time
import math
import os
import sys

from lib import model_utils
from configs import model_config


def main(_):
    config = model_config.Config()
    #TODO: 클래스의 모든 변수를 출력할 수 있는 법
    print("batch size: ", config.batch_size)

    with tf.Session() as sess:
        # if forward_only == False, Training Mode. Batch size is modified, include optimizer.
        forward_only = False

        user_vocab_path = os.path.join(config.data_dir, 'vocab_user.in')
        bot_vocab_path = os.path.join(config.data_dir, 'vocab_bot.in')

        # create model
        model = model_utils.create_model(sess, config, forward_only)

# main()
if __name__ == "__main__":
    tf.app.run()
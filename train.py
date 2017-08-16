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

    # 새로운 tensorflow 세션을 생성합니다 이후 sess 라는 이름으로 호출합니다
    with tf.Session() as sess:
        # forward_only 가 False 일떄 모델을 학습가능한 형태로 생성합니다. 배치 사이즈가 변경되고 모델 그래프에 옵티마이저가 포함 됩니다.
        forward_only = False

        # vacab파일의 경로를 정의합니다. 여러 모델의 테스트를 위해서 사전 크기를 파일명뒤에 명시적으로 표시합니다
        # 예를들어 8000 사이즈의 vovab 파일은 vocab8000.in 입니다.
        user_vocab_path = os.path.join(config.data_dir, 'vocab_user.in')
        bot_vocab_path = os.path.join(config.data_dir, 'vocab_bot.in')


        # create model
        model = model_utils.create_model(sess, config, config.batch_size, forward_only)
        print()


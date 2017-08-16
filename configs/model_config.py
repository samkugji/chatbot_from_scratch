import tensorflow as tf

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


class ModelConfig():
    # Path
    root_dir = "./"
    data_dir = root_dir + "data"
    model_dir = root_dir + "nn_models"
    reply_dir = root_dir + "reply"

    # 모델 학습 시 사용하는 데이터 형식
    data_type = tf.float32

    # for learning
    learning_late = 0.001
    learning_rate_decay_fator = 0.999
    max_gradient_norm = 5.0

    # for model
    vocab_size = 8000
    batch_size = 256
    # for encoding
    enc_hidden_size = 128
    enc_num_layers = 1
    # for decoding
    dec_hidden_size = 128
    dec_num_layers = 1

    max_epoch = 100000
    checkpoint_step = 100

    buckets = [(5,10), (8,15)]



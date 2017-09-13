import tensorflow as tf

# https://www.tensorflow.org/tutorials/seq2seq
# http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
# PAD: padding(Filler)
# GO: prefix of decoder input
# EOS: suffix of decoder output
# UNK: Unknown; word not in vocabulary
# Q : [ PAD, PAD, PAD, PAD, PAD, PAD, “?”, “you”, “are”, “How” ]  # reversing input is replaced by attention
# A : [ GO, “I”, “am”, “fine”, “.”, EOS, PAD, PAD, PAD, PAD ]
# If we are using the bucket (5,10), our sentences will be encoded to :
# Q : [ PAD, “?”, “you”, “are”, “How” ]
# A : [ GO, “I”, “am”, “fine”, “.”, EOS, PAD, PAD, PAD, PAD ]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


class Config():
    # Path
    root_dir = "./"
    data_dir = root_dir + "data"
    model_dir = root_dir + "nn_models"
    reply_dir = root_dir + "reply"

    # 모델 학습 시 사용하는 데이터 형식
    data_type = tf.float32

    # for model
    vocab_size = 8000
    batch_size = 256
    use_lstm = True

    # for encoding
    enc_hidden_size = 128
    enc_num_layers = 1
    # for decoding
    dec_hidden_size = 128
    dec_num_layers = 1

    # for learning
    learning_rate = 0.001
    learning_rate_decay_fator = 0.999
    max_gradient_norm = 5.0

    # bucket <- (encoder input = user query , decoder input = bot answer)
    buckets = [(5,10), (8,15)]

    max_epoch = 100000
    checkpoint_step = 100
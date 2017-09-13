import sys
import numpy as np
import codecs

from configs import model_config


def get_batch(chat_token_ids, config):
    #np.random.randint(low, high=None, size=None, dtype='l')
    random_idx = np.random.randint(0, len(chat_token_ids), config.batch_size)

    encoder_inputs = []
    decoder_inputs = []
    encoder_inputs_length = []
    decoder_inputs_length = []
    target_weights = []

    for i in range(config.batch_size):
        encoder_inputs.append(chat_token_ids[random_idx[i]][0])
        decoder_inputs.append(chat_token_ids[random_idx[i]][1])
        encoder_inputs_length.append(len(chat_token_ids[random_idx[i]][0]))
        decoder_inputs_length.append(len(chat_token_ids[random_idx[i]][1]))

    max_encoder_length = max(encoder_inputs_length)
    max_decoder_length = max(decoder_inputs_length)

    for i in range(config.batch_size):
        # do padding
        temp_encoder = encoder_inputs[i]



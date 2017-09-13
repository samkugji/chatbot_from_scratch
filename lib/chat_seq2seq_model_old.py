import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell, MultiRNNCell

from configs import model_config

class ChatSeq2SeqModel(object):
    """
    Multi-layer seq2seq with attention network + (highway connections)
    """

    def __init__(self, config, forward_only=False):
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.use_lstm = config.batch_size
        self.enc_hidden_size = config.enc_hidden_size
        self.enc_num_layers = config.enc_num_layers
        self.dec_hidden_size = config.dec_hidden_size
        self.dec_num_layers = config.dec_num_layers

        # declaration as tf.Variable for dynamic learning rate
        # https://www.tensorflow.org/programmers_guide/variables
        # self.learning_rate = tf.Variable(float(config.learning_late), trainable=False)
        self.learning_rate = tf.get_variable(name="learning_rate",
                                             initializer=float(config.learning_rate),
                                             trainable=False
                                             )

        self.learning_rate_decay_fator = self.learning_rate.assign(
            self.learning_rate * config.learning_rate_decay_fator
        )
        self.global_step = tf.get_variable(name="global_step",
                                           initializer=0
                                           )

        self.max_gradient_norm = config.max_gradient_norm
        self.buckets = config.buckets

        # RNN cells for encoding layer 1(bi-LSTM or bi-GRU)
        if self.use_lstm:
            single_cell_1 = LSTMCell(self.enc_hidden_size)
            single_cell_2 = LSTMCell(self.dec_hidden_size)
        else:
            single_cell_1 = GRUCell(self.enc_hidden_size)
            single_cell_2 = GRUCell(self.dec_hidden_size)

        # multi layers
        enc_cell = MultiRNNCell([single_cell_1 for _ in range(self.enc_num_layers)])
        dec_cell = MultiRNNCell([single_cell_2 for _ in range(self.dec_num_layers)])

        self.encoder_cell = enc_cell
        self.decoder_cell = dec_cell

        self._make_graph(forward_only)

        self.saver = tf.train.Saver(tf.global_variables())

    def _make_graph(self, forward_only):

        self._init_data()
        self._init_embeddings()
        self._init_encoder()
        self._init_decoder(forward_only)

        if not forward_only:
            self._init_optimizer()

        # for tensorboard
        writer = tf.summary.FileWriter("/tmp/test_logs", self.sess.graph)


    def _init_data(self):
        # ex) encoder_inputs's placeholder <- 2-dimension matrix
        # [
        #   [36, 6, 36, 6, 14, 5, 13, 35, 739, 41, 24, 103, EOS_ID],
        #   [3, 5, 13, 956, 3, 227, EOS_ID, GO_ID, 142, 331, 4, 17, 8, 112, 6, 155, 3, EOS_ID] , ...
        # ]
        self.encoder_inputs = tf.placeholder(shape=(None, None),
                                            dtype=tf.int32,
                                            name="encoder_inputs")
        self.encoder_inputs_length = tf.placeholder(shape=(None,),
                                            dtype=tf.int32,
                                            name="encoder_inputs_length")

        # ex) decoder_inputs's placeholder <- 2-dimension matrix
        # [
        #   [GO_ID, 5, 15, 33, 12, 2021, 3,2274,EOS_ID],
        #   [GO_ID, 142, 331, 4, 17, 8, 112, 6, 155, 3, EOS_ID] , ...
        # ]
        self.decoder_inputs = tf.placeholder(shape=(None, None),
                                            dtype=tf.int32,
                                            name="decoder_inputs")
        self.decoder_inputs_length = tf.placeholder(shape=(None,),
                                            dtype=tf.int32,
                                            name="decoder_inputs_length")
        # get rid of 'GO_ID' from self.decoder_inputs
        self.decoder_target = self.decoder_inputs[1:, :]

        # fit data into buckets
        self.encoder_inputs_bucket = self.encoder_inputs[:self.buckets[-1][0], :]
        self.decoder_inputs_bucket = self.decoder_inputs[:self.buckets[-1][1], :]

        # for cross-entropy, decoder_input 패딩된 데이터와 실제 데이터를 골라내기 위한 mask 입니다.
        # getbatch 에서 받은 데이터를 주입할 placeholder
        self.target_weights = tf.placeholder(shape=(self.batch_size, None), dtype=tf.float32, name="target_weights")
        self.prev_hidden = self.encoder_cell.zero_state(self.batch_size, dtype=tf.float32)

    def _init_embeddings(self):
        with tf.variable_scope("embbeding") as scope:

            # make embedding_matrix(vocab_size , hidden_size)
            self.enc_embedding_matrix = tf.get_variable(
                name="enc_embbeding_matrix",
                shape=[self.vocab_size, self.enc_hidden_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32
            )

            self.dec_embedding_matrix = tf.get_variable(
                name="dec_embbeding_matrix",
                shape=[self.vocab_size, self.dec_hidden_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32
            )

            # 연속된 단어의 index값으로 표현된 입력값을 각 인덱스의 ont-hot으로 표현하고 이어서
            # embedding_vector화 하는 과정을 embedding_lookup을 통해서 처리
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.enc_embedding_matrix, self.encoder_inputs_bucket)

            self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                self.dec_embedding_matrix, self.decoder_inputs_bucket)

    def _init_encoder(self):
        with tf.variable_scope("encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = tf.nn.dynamic_rnn(
                cell=self.encoder_cell,
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                time_major=True,
                dtype=tf.float32
            )

    def _init_decoder(self, forward_only):
        with tf.variable_scope("decoder") as scope:

            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            # attention_states: size [batch_size, max_time, num_units]
            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

            # encoder_outputs를 가지고 attention network에 필요한 값 생성
            (attention_keys, attention_values, attention_score_fn, attention_construct_fn) = (
                seq2seq._prepare_attention(
                    attention_states=attention_states,
                    attention_op="bahdanau",
                    num_units=self.dec_hidden_size
                )
            )

            # for only prediction
            if forward_only:
                # decoder 함수로 inference를 사용하고
                # 아래의 train 과정보다 더 많은 인자를 입력 받는데
                # 전처리 과정에서 진행한 embedding 과정의 역순을 일부 자동으로 처리하기 위함입니다.
                decoder_fn = seq2seq.attention_decoder_fn_inference(
                    output_fn = output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.dec_embedding_matrix,
                    start_of_sequence_id=model_config.GO_ID,
                    end_of_sequence_id=model_config.EOS_ID,
                    maximum_length=self.buckets[-1][1],
                    num_decoder_symbols=self.vocab_size,
                )

                # rnn_decoder layer 생성성
                # encoder지나서 계산된 encdoer_state는 decoder_fn의 인자를 통해 decoder에 연결
                (self.decoder_outputs, self.decoder_state, self.decoder_context_state) = (
                    seq2seq.dynamic_rnn_decoder(
                        cell=self.decoder_cell,
                        decoder_fn=decoder_fn,
                        time_major=True,
                        scope=scope,
                    )
                )

                # decoder_ouputs에서 logit형태로 예측값 출력
                self.decoder_logits = self.decoder_outputs

            # for only training
            else:
                # decoder 함수로 train을 사용
                # 이전 레이어의 출력값인 encoder_state와 함께 attention에 필요한 값 입력
                decoder_fn = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name="attention_decoder"
                )
                # rnn_decoder layer 생성
                # encoder를 지나서 계산된 encoder_state는 decoder_fn을 인자로 통해 decoder에 연결
                # loss값을 계산하기 위해 decoder_output을 출력
                (self.decoder_outputs, self.decoder_state, self.decoder_context_state) = (
                    seq2seq.dynamic_rnn_decoder(
                        cell=self.decoder_cell,
                        decoder_fn=decoder_fn,
                        inputs=self.decoder_inputs_embedded,
                        sequence_length=self.decoder_inputs_length,
                        time_major=True,
                        scope=scope,
                    )
                )

                self.decoder_logits = output_fn(self.decoder_outputs)

            # vocab사이즈 만큼의 각각 단어의 확률값으로 표현된 리스트의 순서열을 얻고
            # argmax 연산을 통해 최대값 찾기
            self.decoder_prediction = tf.argmax(self.decoder_logits, axis=1, name="decoder_prediction")

            # loss 계산을 위한 logit과 targets을 출력
            logits = tf.transpose(self.decoder_logits, [1, 0, 2])
            targets = tf.transpose(self.decoder_target, [1, 0])

            if not forward_only:
                self.loss = seq2seq.sequence_loss(logits=logits, targets=targets, weights=self.target_weights)

    def _init_optimizer(self):
        params = tf.trainable_variables()

        self.gradient_norms = []
        self.updates = []

        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params)

        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

    def prediction(self, session, encoder_inputs, encoder_inputs_length, forward_only):
        input_feed = {
            self.encoder_inputs: encoder_inputs,
            self.encoder_inputs_length: encoder_inputs_length,
        }

        if forward_only:
            output_feed = self.decoder_prediction
            prediction = session.run(output_feed, input_feed)

            return prediction

    def step(self, session, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length,
             target_weights, forward_only):
        input_feed = {
            self.encoder_inputs: encoder_inputs,
            self.encoder_inputs_length: encoder_inputs_length,
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_length,
            self.target_weights: target_weights
        }
        # 예측모델이면 출력에 필요한 op만 계산
        if forward_only:
            output_feed = [self.decoder_prediction, self.decoder_prediction, self.encoder_state, self.decoder_state]
            prediction = session.run(output_feed, input_feed)

            return prediction

        # 학습모델이면 updates 계산을 통해 학습 진행
        else:
            output_feed = [self.updates, self.gradient_norms, self.loss, self.encoder_state, self.decoder_state]
            updates, gradient, loss, encoder_embedding, decoder_embedding = session.run(output_feed, input_feed)

            return gradient, loss, None, None, encoder_embedding, decoder_embedding
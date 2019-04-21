import tensorflow as tf
from tensorflow.contrib import rnn
from utils import get_init_embedding


class Model(object):
    def __init__(self, reversed_dict, article_max_len, summary_max_len, args, forward_only=False):
        self.vocabulary_size = len(reversed_dict)
        self.embedding_size = args.embedding_size
        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = args.learning_rate
        self.beam_width = args.beam_width
        self.cell_type = 'lstm'
        if not forward_only:
            self.keep_prob = args.keep_prob
        else:
            self.keep_prob = 1.0
        self.cell = tf.nn.rnn_cell.BasicLSTMCell  # 采用lstm cell 方式。
        with tf.variable_scope("decoder/projection"):
            self.projection_layer = tf.layers.Dense(self.vocabulary_size, use_bias=False)

        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.X = tf.placeholder(tf.int32, [None, article_max_len])  ##[batch_size,30]
        self.X_len = tf.placeholder(tf.int32, [None])
        self.decoder_input = tf.placeholder(tf.int32, [None, summary_max_len])  # [batch_size,15]
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, summary_max_len])  # [batch_size,15]
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("embedding"):
            if not forward_only and args.glove:  # 当训练状态，比并且参数选择了 glove打开状态。
                init_embeddings = tf.constant(get_init_embedding(reversed_dict, self.embedding_size), dtype=tf.float32)
            else:
                init_embeddings = tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.encoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.X),
                                                perm=[1, 0, 2])  # (50， batch_size, 300)
            self.decoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.decoder_input),
                                                perm=[1, 0, 2])  # (15, batch_size, 300)

        with tf.name_scope("encoder"):  # 当encoder时， 设置 前项和后项cell的 长度，以及隐藏cell 大小。  设置为 hidden150, 两层。

            #encoder_outputs, encoder_state_fw, encoder_state_bw = self._stack_bi_dynamic_rnn_cell()
            #   输入值，  利用前项和后项cell生成对应的长度的双向lstm，  time_major 是输入的batch_size 对应位置。stack_bidirectional_dynamic_rnn  bidirectional_dynamic_rnn  是两个相似的双心啊lstm
            encoder_outputs, encoder_state = self._bi_dynamic_rnn_cell()
            self.encoder_output = tf.concat(encoder_outputs, 2)  # encoder_outputs: 50,batch,300.   encoder_output
            self.encoder_state = encoder_state

        # 在他程序中 是把 第一层的的 开始状态 当做  decode 的状态信息。 个人对次表示 怀疑。
        with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
            decoder_cell = self.cell(self.num_hidden * 2)  # 从这里可以看出，decode,  只需要单层的解析。 decode_cell 是 300维度

            if not forward_only:
                attention_states = tf.transpose(self.encoder_output, [1, 0, 2])  # batch_size, 50, 300
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    # 初始化attention的状态，隐层cell 300,，encode的所有数据。 每个句子的长度， 是否归一化。
                    self.num_hidden * 2, attention_states, memory_sequence_length=self.X_len,
                    normalize=True)  # 初始化attention的相关参数。
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   # 单层lstm，attention的方法，attention的输出深度。
                                                                   attention_layer_size=self.num_hidden * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32,
                                                        batch_size=self.batch_size)  # 给decode_cell 定义初始化。batch_size 不太懂。  lstmcell, attention,attention, attention,tuple, attention.不过对于如何初始化beam_search 比较特殊。
                initial_state = initial_state.clone(cell_state=self.encoder_state)  # 重新定义它的初始化状态。
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_len,
                                                           time_major=True)  # 输入层定义。  decode的输入值，和它对应的长度，是否是那个奇葩格式。
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                          initial_state)  # cell rnn_cell的特殊形式, helper数据输入, initial_state ： The initial state of the RNNCell., output_layer=None
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True,
                                                                  scope=decoder_scope)  # 真正的执行过程，调用函数运行， 之前都是 初始化某个零件之类的。
                self.decoder_output = outputs.rnn_output
                self.logits = tf.transpose(  # self.decoder_output 的数据为 （ ， ，300） 变为 （ ，，17211） 最后生成的句子长度不确定了。
                    self.projection_layer(self.decoder_output), perm=[1, 0, 2])
                self.logits_reshape = tf.concat(  # 两个数据 在第一维度 融合在一起，让长度保持一致。
                    [self.logits,
                     tf.zeros([self.batch_size, summary_max_len - tf.shape(self.logits)[1], self.vocabulary_size])],
                    axis=1)
            else:
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    # encoder_output:（50，batch_size, 300） 返回值为 batch_size*beam_width, 50,300
                    tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state,
                                                                          multiplier=self.beam_width)  # (,300) 变为 某个值*10，300
                tiled_seq_len = tf.contrib.seq2seq.tile_batch(self.X_len, multiplier=self.beam_width)  # x的长度 也进行相同的操作。
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.num_hidden * 2, tiled_encoder_output, memory_sequence_length=tiled_seq_len, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
                initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.projection_layer
                )
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True, maximum_iterations=summary_max_len, scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

        with tf.name_scope("loss"):
            if not forward_only:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_reshape, labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_len, summary_max_len, dtype=tf.float32)
                self.loss = tf.reduce_sum(crossent * weights / tf.to_float(self.batch_size))

                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def _rnn_cell(self):
        if self.cell_type == 'lstm':
            self.cell = rnn.BasicLSTMCell
        elif self.cell_type == 'gru':
            self.cell = rnn.GRUCell
        fw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
        bw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
        fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in
                    fw_cells]  # 在本实验中 dropout 没有设置，是否会出现过拟合，暂时不确定。
        bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in bw_cells]
        return fw_cells, bw_cells

    def _stack_bi_dynamic_rnn_cell(self):
        """
        双向RNN
        需要 cell_type , num_layers,  num_hidden,  dropout_rate
        :return:
        """
        fw_cells, bw_cells = self._rnn_cell()
        #   输入值，  利用前项和后项cell生成对应的长度的双向lstm，  time_major 是输入的batch_size 对应位置。
        #   stack_bidirectional_dynamic_rnn  bidirectional_dynamic_rnn  是两个相似的双心啊lstm
        encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            fw_cells, bw_cells, self.encoder_emb_inp,
            sequence_length=self.X_len, time_major=True, dtype=tf.float32)

        encoder_state_c = tf.concat((encoder_state_fw[-1].c, encoder_state_bw[-1].c),
                                    1)  # 这里有点 反逻辑，不过想清楚，就知道了。 获得双向lstm的首位置。
        encoder_state_h = tf.concat((encoder_state_fw[-1].h, encoder_state_bw[-1].h), 1)  # 此处扩展代码中 可以进一步加深认识。
        encoder_states = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        return encoder_outputs, encoder_states

    def _bi_dynamic_rnn_cell(self):

        fw_cells, bw_cells = self._rnn_cell()

        fw_cells = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_cells = tf.contrib.rnn.MultiRNNCell(bw_cells)

        encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, self.encoder_emb_inp,
                                                                         sequence_length=self.X_len, time_major=True,
                                                                         dtype=tf.float32)
        encoder_state_c = tf.concat([encoder_state[0][-1].c, encoder_state[1][-1].c], -1)
        encoder_state_h = tf.concat([encoder_state[0][-1].h, encoder_state[1][-1].h], -1)

        encoder_states = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        return encoder_outputs, encoder_states

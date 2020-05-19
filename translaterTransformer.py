import tensorflow as tf
import pdb


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A t
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)  # 计算 inputs的均值和方差
        beta = tf.Variable(tf.zeros(params_shape))
        gama = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))

        outputs = gama * normalized + beta
        return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable(name='lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def multihead_attention(key_emb,
                        que_emb,
                        queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        #         pdb.set_trace()

        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(key_emb, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(que_emb, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''

    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


class TransformerModel():
    def __init__(self, is_training=True):
        tf.reset_default_graph()
        self.hidden_units = arg.hidden_units
        self.input_vocab_size = arg.input_vocab_size
        self.label_vocab_size = arg.label_vocab_size
        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.max_length = arg.max_length
        self.lr = arg.lr
        self.dropout = arg.dropout_rate

        # input
        self.x = tf.placeholder(tf.int32, shape=(None, None))
        self.y = tf.placeholder(tf.int32, shape=(None, None))
        self.de_inp = tf.placeholder(tf.int32, shape=(None, None))
        self.is_training = tf.placeholder(tf.bool)

        # Encoder
        with tf.variable_scope("encoder"):
            # embedding
            self.en_emb = embedding(self.x, vocab_size=self.input_vocab_size, num_units=self.hidden_units, scale=True,
                                    scope="enc_embed")
            self.enc = self.en_emb + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                vocab_size=self.max_length, num_units=self.hidden_units, zero_pad=False, scale=False, scope="enc_pe")
            #             pdb.set_trace()

            # Dropout
            self.enc = tf.layers.dropout(self.enc, rate=self.dropout, training=tf.convert_to_tensor(self.is_training))

            # Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks-{}".format(i)):
                    # Multihead Attention
                    self.enc = multihead_attention(key_emb=self.en_emb,
                                                   que_emb=self.en_emb,
                                                   queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=self.hidden_units,
                                                   dropout_rate=self.dropout,
                                                   is_training=self.is_training,
                                                   causality=False)

            # Feed Forward
            self.enc = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])

        # Decoder
        with tf.variable_scope("decoder"):
            # embedding
            self.de_emb = embedding(self.de_inp, vocab_size=self.label_vocab_size, num_units=self.hidden_units,
                                    scale=True, scope="dec_embed")
            self.dec = self.de_emb + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.de_inp)[1]), 0), [tf.shape(self.de_inp)[0], 1]),
                vocab_size=self.max_length, num_units=self.hidden_units, zero_pad=False, scale=False, scope="dec_pe")

            ## Multihead Attention ( self-attention)
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.dec = multihead_attention(key_emb=self.de_emb,
                                                   que_emb=self.de_emb,
                                                   queries=self.dec,
                                                   keys=self.dec,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope='self_attention')

            ## Multihead Attention ( vanilla attention)
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.dec = multihead_attention(key_emb=self.en_emb,
                                                   que_emb=self.de_emb,
                                                   queries=self.dec,
                                                   keys=self.enc,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope='vanilla_attention')

                    ### Feed Forward
            self.outputs = feedforward(self.dec, num_units=[4 * self.hidden_units, self.hidden_units])

        # 最终线性投影
        self.logits = tf.layers.dense(inputs=self.outputs, units=self.label_vocab_size, activation=None)
        self.preds = tf.to_int32(tf.arg_max(self.logits, -1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
            tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)

        if is_training:
            # loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

            # Training Scheme
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()


from tqdm import tqdm
import json

inputs = []
outputs = []
with open(file="H:\\PycharmProjects\\dataset\\translation2019zh\\translation2019zh_train.json", mode="r",
          encoding="utf-8") as f:
    for line in tqdm(f.readlines()):
        dic = json.loads(line)
        inputs.append(dic['english'].replace(',', ' ,').lower())
        outputs.append(dic['chinese'])

train_data_num_start = 110000
train_data_num_end = train_data_num_start + 90000

inputs = inputs[train_data_num_start: train_data_num_end]
outputs = outputs[train_data_num_start: train_data_num_end]
print(inputs[:10])
print(outputs[:10])


inputs = [en.split(' ') for en in tqdm(inputs) if en != ',']
print(inputs[:10])


import jieba
outputs = [[char for char in jieba.cut(line) if char != ' '] for line in tqdm(outputs)]
print(outputs[:10])

import os
def get_vocab_cache(fname='encoder_vocab_mapping.txt'):
    data = []
    if(os.path.exists(fname)):
        with open(fname, mode='r', encoding='utf-8') as r:
            data = r.read().split(' ')
    return data

def save_vocab_cache(data, fname='encoder_vocab_mapping.txt'):
    with open(fname, mode='w', encoding='utf-8') as w:
        for word in data:
            w.write(word + ' ')
        w.close()


import numpy as np
import os

SOURCE_CODES = ['<PAD>']
TARGET_CODES = ['<PAD>', '<GO>', '<EOS>']


def get_vocab(data, init=['<PAD>'], cache_fname='encoder_vocab_mapping.txt'):
    vocab = init
    cache = get_vocab_cache(cache_fname)
    if (len(cache) > 0):
        vocab = cache

    for line in tqdm(data):
        for word in line:
            if word not in vocab:
                vocab.append(word)
    return vocab


encoder_vocab = get_vocab(inputs, init=SOURCE_CODES, cache_fname='encoder_vocab_mapping.txt')
decoder_vocab = get_vocab(outputs, init=TARGET_CODES, cache_fname='decoder_vocab_mapping.txt')
print(encoder_vocab[:10])
print(decoder_vocab[-20:-10])


encoder_inputs = [[encoder_vocab.index(word) for word in line] for line in tqdm(inputs)]
decoder_val_used = [[decoder_vocab.index(word) for word in line] for line in tqdm(outputs)]
decoder_inputs = [[decoder_vocab.index('<GO>')] + mapping_index for mapping_index in tqdm(decoder_val_used)]
decoder_targets = [mapping_index + [decoder_vocab.index('<EOS>')] for mapping_index in tqdm(decoder_val_used)]
print(decoder_inputs[:20])
print(decoder_targets[:20])

from keras.preprocessing.sequence import pad_sequences


def get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size, batch_num):
    batch_num -= 1
    for k in (range(batch_num)):
        start = k * batch_size
        end = start + batch_size

        en_inputs_batch = encoder_inputs[start: end]
        de_inputs_batch = decoder_inputs[start: end]
        de_targets_batch = decoder_targets[start: end]

        en_len = [len(line) for line in en_inputs_batch]
        de_len = [len(line) for line in de_inputs_batch]

        #         if len(en_len) == 0 or len(de_len) == 0:
        #             continue

        max_words_len = 64
        max_en_len = max_words_len if max(en_len) > max_words_len else max(en_len)
        max_de_len = max_words_len if max(de_len) > max_words_len else max(de_len)

        en_inputs_batch = pad_sequences(sequences=en_inputs_batch, maxlen=max_en_len, padding='post', dtype='int32',
                                        value=0)
        de_inputs_batch = pad_sequences(sequences=de_inputs_batch, maxlen=max_de_len, padding='post', dtype='int32',
                                        value=0)
        de_targets_batch = pad_sequences(sequences=de_targets_batch, maxlen=max_de_len, padding='post', dtype='int32',
                                         value=0)
        yield en_inputs_batch, de_inputs_batch, de_targets_batch


batch_size = 4
batch = get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size, len(encoder_inputs) // batch_size)
# next(batch)

def create_hparams():
    params = tf.contrib.training.HParams(
        num_heads = 8,
        num_blocks = 6,
        # vocab
        input_vocab_size = 50,
        label_vocab_size = 50,
        # embedding size
        max_length = 100,
        hidden_units = 128,
        dropout_rate = 0.0,
        lr = 0.000003)
    return params

arg = create_hparams()
arg.input_vocab_size = len(encoder_vocab)
arg.label_vocab_size = len(decoder_vocab)

from sklearn.utils import shuffle
validate_ratio = 0.2

encoder_inputs, decoder_inputs, decoder_targets = shuffle(encoder_inputs, decoder_inputs, decoder_targets)
train_num = int(len(encoder_inputs) * (1 - validate_ratio))
train_encoder_inputs = encoder_inputs[:train_num]
val_encoder_inputs = encoder_inputs[train_num:]
train_decoder_inputs = decoder_inputs[:train_num]
val_decoder_inputs = decoder_inputs[train_num:]
train_decoder_targets = decoder_targets[:train_num]
val_decoder_targets = decoder_targets[train_num:]

print(train_encoder_inputs[:3])
print(train_decoder_inputs[:3])
print(train_decoder_targets[:3])

import tensorflow as tf
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 程序按需申请内存

epochs = 25
batch_size = 4

model = TransformerModel()

saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
with tf.Session(config=config) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    model_path = 'logs'
    if (os.path.exists(os.path.join(model_path, 'TransformerModel.meta'))):
        saver.restore(sess, "logs\TransformerModel")

    writer = tf.summary.FileWriter('tensorboard/train', tf.get_default_graph())
    validation_writer = tf.summary.FileWriter('tensorboard/validate', tf.get_default_graph())

    for k in range(epochs):
        total_loss = 0
        batch_num = len(train_encoder_inputs) // batch_size
        batch = get_batch(train_encoder_inputs, train_decoder_inputs, train_decoder_targets, batch_size, batch_num)
        val_batch_num = len(val_encoder_inputs) // batch_size
        val_batch = get_batch(val_encoder_inputs, val_decoder_inputs, val_decoder_targets, batch_size, val_batch_num)

        start_time = time.time()
        for i in range(batch_num):
            try:
                encoder_input, decoder_input, decoder_target = next(batch)
            except StopIteration:
                break
            feed = {model.x: encoder_input, model.y: decoder_input, model.de_inp: decoder_target,
                    model.is_training: True}
            cost, _, acc = sess.run([model.mean_loss, model.train_op, model.acc], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                val_encoder_input, val_decoder_input, val_decoder_target = next(val_batch)
                val_feed = {model.x: val_encoder_input, model.y: val_decoder_input, model.de_inp: val_decoder_target,
                            model.is_training: False}
                val_acc = sess.run([model.acc], feed_dict=val_feed)

                print("batch num :", i, ", train average loss:", total_loss / (i + 1), ", acc:", acc)
                print("validation current acc:", val_acc[0])

                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)

                vs = sess.run(merged, feed_dict=val_feed)
                validation_writer.add_summary(vs, k * batch_num + i)

        end_time = time.time()
        print('one epoch consume time:', (end_time - start_time), "s")
        print('=== epoch:', k + 1, ' average loss: ', total_loss / (batch_num + 1), " accuracy:", acc)
        print('-----**-----**-----**-----**-----**-----**-----**-----**-----**-----**-----')
        saver.save(sess=sess, save_path=os.path.join(model_path, 'TransformerModel'))

    writer.close()
    validation_writer.close()

from tqdm import tqdm
import json
import jieba

test_inputs = []
test_outputs = []
with open(file="H:\\PycharmProjects\\dataset\\translation2019zh\\translation2019zh_valid.json", mode="r",
          encoding="utf-8") as f:
    for line in tqdm(f.readlines()):
        dic = json.loads(line)
        test_inputs.append(dic['english'].replace(',', ' ,').lower())
        test_outputs.append(dic['chinese'])

test_data_ratio = 0.0125
test_inputs_ori, test_outputs_ori = shuffle(test_inputs, test_outputs)
test_data_num = int(len(test_inputs_ori) * test_data_ratio)
test_inputs = test_inputs_ori[:test_data_num]
test_outputs = test_outputs_ori[:test_data_num]
test_inputs = [en.split(' ') for en in tqdm(test_inputs) if en != ',']
test_outputs = [[char for char in jieba.cut(line) if char != ' '] for line in tqdm(test_outputs)]
test_encoder_vocab = get_vocab(test_inputs, init=SOURCE_CODES, cache_fname='encoder_vocab_mapping.txt')
test_decoder_vocab = get_vocab(test_outputs, init=TARGET_CODES, cache_fname='decoder_vocab_mapping.txt')
test_encoder_inputs = [[encoder_vocab.index(word) for word in line] for line in tqdm(test_inputs)]
test_decoder_val_used = [[test_decoder_vocab.index(word) for word in line] for line in tqdm(test_outputs)]
test_decoder_inputs = [[test_decoder_vocab.index('<GO>')] + mapping_index for mapping_index in
                       tqdm(test_decoder_val_used)]
test_decoder_targets = [mapping_index + [test_decoder_vocab.index('<EOS>')] for mapping_index in
                        tqdm(test_decoder_val_used)]


def get_test_batch(encoder_inputs, decoder_inputs, decoder_targets):
    for i in range(len(encoder_inputs)):
        yield np.array([encoder_inputs[i]]), np.array([decoder_inputs[i]]), np.array([decoder_targets[i]])

test_batch = get_test_batch(test_encoder_inputs, test_decoder_inputs, test_decoder_targets)
next(test_batch)

model = TransformerModel()

saver = tf.train.Saver()
model_path = 'logs'

with tf.Session() as sess:
    if (os.path.exists(os.path.join(model_path, 'TransformerModel.meta'))):
        saver.restore(sess, "logs\TransformerModel")

    test_acc_count = 0.0
    for j in range(len(test_encoder_inputs)):
        try:
            test_encoder_input, test_decoder_input, test_decoder_target = next(test_batch)
        except StopIteration:
            break

        de_inp = [[test_decoder_vocab.index('<GO>')]]
        test_feed = {model.x: test_encoder_input, model.de_inp: np.array(de_inp), model.is_training: False}
        preds = sess.run(model.preds, feed_dict=test_feed)
        #         pdb.set_trace()
        if preds[0][-1] == test_decoder_vocab.index('<EOS>'):
            break
        de_inp[0].append(preds[0][-1])
        decoder_preds_words = ''.join([test_decoder_vocab[index] for index in de_inp[0][1:]])

        print("*****===*****===*****===*****===*****===*****===*****===*****")
        print("要求翻译的句子：", test_inputs_ori[j])
        print("预测翻译成：", decoder_preds_words)
        print("正确翻译为：", test_outputs_ori[j])
        print("*****===*****===*****===*****===*****===*****===*****===***** \n")
        decoder_preds_words = ''


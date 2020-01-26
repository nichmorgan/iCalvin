from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

tf.random.set_seed(1234)

import matplotlib.pyplot as plt
import unicodedata
import re
import pandas


class Dataset:
    def __init__(self, path, ascii=True, max_samples=5000, max_length=40,
                 batch_size=64, buffer_size=2000):

        self.max_samples = max_samples
        self.max_length = max_length
        self.path = path
        self.ascii = ascii

        self.questions, self.answers = self._load_conversations()
        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            self.questions + self.answers, target_vocab_size=2 ** 13
        )
        self.start_token, self.end_token = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
        self.vocab_size = self.tokenizer.vocab_size + 2

        questions, answers = self._tokenize_and_filter(self.questions, self.answers)

        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': questions,
                'dec_inputs': answers[:, :-1]
            },
            {
                'outputs': answers[:, 1:]
            }
        ))
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self._dataset = dataset

    def _unicode_to_ascii(self, s):
        s_ascii = ''
        for c in unicodedata.normalize('NFD', s):
            if unicodedata.category(c) != 'Mn':
                s_ascii += c

        return s_ascii

    def _preprocess_sentence(self, sentence):
        if self.ascii:
            sentence = self._unicode_to_ascii(sentence.lower().strip())
        sentence = re.sub(r'([-?.!,])', r' \1 ', sentence)
        sentence = re.sub(r'[" "]+', ' ', sentence)
        sentence = re.sub(r'[^\w?.!,]+', ' ', sentence)

        sentence = sentence.strip()

        return sentence

    def posprocess_sentence(self, sentence):
        sentence = re.sub(r' ([?.!,])', r'\1 ', sentence)
        sentence = re.sub(r' ([-]) ', r'\1', sentence)
        sentence = re.sub(r'[" "]+', ' ', sentence).strip()

        return sentence

    def _load_conversations(self):
        df = pandas.read_csv(self.path, usecols=['pergunta', 'resposta'], encoding='utf-8')

        for column in df.columns:
            df.loc[:, column] = df[column].apply(self._preprocess_sentence)[:self.max_samples]

        return df['pergunta'].values, df['resposta'].values

    def _tokenize_and_filter(self, inputs, outputs):
        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(inputs, outputs):
            sentence1 = self.start_token + self.tokenizer.encode(sentence1) + self.end_token
            sentence2 = self.start_token + self.tokenizer.encode(sentence2) + self.end_token

            if len(sentence1) <= self.max_length and len(sentence2) <= self.max_length:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)

        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.max_length, padding='post'
        )
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.max_length, padding='post'
        )

        return tokenized_inputs, tokenized_outputs

    @property
    def dataset(self):
        return self._dataset


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask):
        """Calculate the attention weights. """
        matmul_qk = tf.matmul(query, key, transpose_b=True)

        # scale matmul_qk
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        # add the mask to zero out padding tokens
        if mask is not None:
            logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(logits, axis=-1)

        output = tf.matmul(attention_weights, value)

        return output

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class Encoder:
    def __init__(self, units, d_model, num_heads, dropout,
                 vocab_size, num_layers, name='encoder'):

        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.name = name
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self._encoder = None

    def _create_encoder_layer(self, name):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        attention = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
        attention = tf.keras.layers.Dropout(rate=self.dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(inputs + attention)

        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def _create_encoder(self):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self.vocab_size, self.d_model)(embeddings)

        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(self.num_layers):
            outputs = self._create_encoder_layer(name=f'{self.name}_layer_{i}')([outputs, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=self.name)

    @property
    def encoder(self):
        if not self._encoder:
            self._encoder = self._create_encoder()
        return self._encoder

    def plot_model(self,
                   to_file='encoder.png', show_shapes=True, show_layers_names=True,
                   rankdir='TB', expand_nested=False, dpi=96):

        tf.keras.utils.plot_model(self.encoder, to_file, show_shapes,
                                  show_layers_names, rankdir, expand_nested, dpi)

    def __call__(self, *args, **kwargs):
        return self.encoder(**kwargs)


class Decoder:
    def __init__(self, units, d_model, num_heads, dropout,
                 vocab_size, num_layers, name='decoder'):
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.name = name
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self._decoder = None

    def _create_decoder_layer(self, name):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name="encoder_outputs")
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        attention1 = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention_1")(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
        attention1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention1 + inputs)

        attention2 = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
        attention2 = tf.keras.layers.Dropout(rate=self.dropout)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention2 + attention1)

        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def _create_decoder(self):
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name='encoder_outputs')
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name='look_ahead_mask')
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self.vocab_size, self.d_model)(embeddings)

        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(self.num_layers):
            outputs = self._create_decoder_layer(name=f'{self.name}_layer_{i}')(inputs=[outputs, enc_outputs,
                                                                                        look_ahead_mask, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=self.name)

    @property
    def decoder(self):
        if not self._decoder:
            self._decoder = self._create_decoder()
        return self._decoder

    def plot_model(self,
                   to_file='decoder.png', show_shapes=True, show_layers_names=True,
                   rankdir='TB', expand_nested=False, dpi=96):

        tf.keras.utils.plot_model(self.decoder, to_file, show_shapes,
                                  show_layers_names, rankdir, expand_nested, dpi)

    def __call__(self, *args, **kwargs):
        return self.decoder(**kwargs)


class Transformer:
    def __init__(self, units, d_model, num_heads, dropout,
                 vocab_size, num_layers, name='transformer'):
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.name = name

        self.encoder = Encoder(units, d_model, num_heads, dropout, vocab_size, num_layers)
        self.decoder = Decoder(units, d_model, num_heads, dropout, vocab_size, num_layers)

        self._transformer = None

    def _create_transformer(self):
        def create_padding_mask(x):
            mask = tf.cast(tf.math.equal(x, 0), tf.float32)
            return mask[:, tf.newaxis, tf.newaxis, :]

        def create_look_ahead_mask(x):
            seq_len = tf.shape(x)[1]
            look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            return tf.maximum(look_ahead_mask, create_padding_mask(x))

        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(
            create_look_ahead_mask,
            output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        enc_outputs = self.encoder(inputs=[inputs, enc_padding_mask])

        dec_outputs = self.decoder(inputs=[dec_inputs, enc_outputs,
                                           look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=self.vocab_size, name="outputs")(dec_outputs)

        return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=self.name)

    @property
    def transformer(self):
        if not self._transformer:
            self._transformer = self._create_transformer()
        return self._transformer

    def plot_model(self,
                   to_file='transformer.png', show_shapes=True, show_layers_names=True,
                   rankdir='TB', expand_nested=False, dpi=96):
        tf.keras.utils.plot_model(self.transformer, to_file, show_shapes,
                                  show_layers_names, rankdir, expand_nested, dpi)

    def __call__(self, *args, **kwargs):
        return self.transformer(**kwargs)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def plot(self, step=200000, ylabel='Learning Rate', xlabel='Train Step'):
        step = tf.range(step, dtype=tf.float32)
        plt.plot(self.__call__(step))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Model:
    def __init__(self, dataset, units, d_model, num_heads, dropout, num_layers):

        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers

        self._learning_rate = CustomSchedule(d_model)
        self._optimizer = tf.keras.optimizers.Adam(self._learning_rate, beta_1=.9, beta_2=.98, epsilon=1e-9)

        self.dataset = dataset
        self._model = Transformer(self.units, self.d_model, self.num_heads, self.dropout,
                                  self.dataset.vocab_size, self.num_layers)

    def _loss_function(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.dataset.max_length - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def _accuracy(self, y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, self.dataset.max_length - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    def compile(self):
        self._model.transformer.compile(optimizer=self._optimizer,
                            loss=self._loss_function,
                            metrics=[self._accuracy])

    def fit(self, epochs, weights_file=None, save_weights_file=None):
        if weights_file:
            self.load_weights(weights_file)

        self._model.transformer.fit(self.dataset.dataset, epochs=epochs)

        if save_weights_file:
            self.save_weights(save_weights_file)

    def load_weights(self, path):
        self._model.transformer.load_weights(path)

    def save_weights(self, path):
        self._model.transformer.save_weights(path)

    def evaluate(self, sentence):
        sentence = self.dataset._preprocess_sentence(sentence)
        sentence = tf.expand_dims(
            self.dataset.start_token + self.dataset.tokenizer.encode(sentence) + self.dataset.end_token, axis=0)

        output = tf.expand_dims(self.dataset.start_token, 0)

        for i in range(self.dataset.max_length):
            predictions = self._model(inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.dataset.end_token[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def predict(self, sentence):
        prediction = self.evaluate(sentence)
        predicted_sentence = self.dataset.tokenizer.decode(
            [i for i in prediction if i < self.dataset.tokenizer.vocab_size])

        predicted_sentence = self.dataset.posprocess_sentence(predicted_sentence)

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        if type(dataset) == str:
            self._dataset = Dataset(dataset)
        elif type(dataset) == Dataset:
            self._dataset = dataset
        else:
            raise Exception('Invalid dataset type! Set another.')


DATASET = Dataset('IA/data/dataset.csv', ascii=False)
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
EPOCHS = 500

model = Model(dataset=DATASET,
              units=UNITS,
              d_model=D_MODEL,
              num_heads=NUM_HEADS,
              dropout=DROPOUT,
              num_layers=NUM_LAYERS)

model.compile()
model.load_weights('IA/test_com_acento.hdf5')

# # model.fit(EPOCHS, save_weights_file='test_com_acento.hdf5')
# # model.load_weights('test.hdf5')
# output = model.predict('Qual o fim útil do homem?')

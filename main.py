import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
import innvestigate
import time
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
from innvestigate.utils.tests.networks import base as network_base

data_df = pd.read_csv('data/train.csv')
data_df = data_df[data_df.sentiment != 'neutral']
split = int(0.7 * data_df.shape[0])

train, val = np.split(data_df.sample(frac=1), [split])

train_texts = train.text.values.astype(str)
val_texts = val.text.values.astype(str)

tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(train_texts)

# Convert string to index
train_sequences = tk.texts_to_sequences(train_texts)
val_texts = tk.texts_to_sequences(val_texts)

# Padding
train_data = pad_sequences(train_sequences, maxlen=200, padding='post')
val_data = pad_sequences(val_texts, maxlen=200, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')
val__data = np.array(val_data, dtype='float32')


# =======================Get classes================
def sentiment_to_class(s):
    if s == 'negative':
        return 0
    elif s == 'positive':
        return 1
    elif s == 'neutral':
        return 2


train_classes = train.sentiment.values
train_class_list = [sentiment_to_class(x) for x in train_classes]

val_classes = val.sentiment.values
val_class_list = [sentiment_to_class(x) for x in val_classes]

train_classes = to_categorical(train_class_list)
val_classes = to_categorical(val_class_list)

# =====================Char CNN=======================
# parameter
input_size = 200
vocab_size = len(tk.word_index)
conv_layers = [[128, 7, 3],
               [128, 7, 3],
               [128, 3, -1],
               [64, 3, -1],
               [64, 3, -1],
               [64, 3, 3]]

fully_connected_layers = [128, 128]
num_of_classes = 2
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'

# Embedding weights
embedding_weights = [np.zeros(vocab_size)]

for char, i in tk.word_index.items():
    onehot = np.zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)
print('Load')
embedding_size = embedding_weights.shape[1]

# Embedding layer Initialization
embedding_layer = Embedding(vocab_size + 1,
                            embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])

# Model Construction
# Input

inputs = Input(shape=(input_size,), name='input', dtype='int64')
# Embedding
emb = embedding_layer(inputs)
# Conv
layer1 = Conv1D(conv_layers[:1][0][0], conv_layers[:1][0][1])(emb)
x = Activation('relu')(layer1)
x = MaxPooling1D(pool_size=conv_layers[:1][0][2])(x)

for filter_num, filter_size, pooling_size in conv_layers[1:]:
    x = Conv1D(filter_num, filter_size)(x)
    x = Activation('relu')(x)
    if pooling_size != -1:
        x = MaxPooling1D(pool_size=pooling_size)(x)
x = Flatten()(x)  # (None, 8704)
# Fully connected layers
for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation='relu')(x)
    x = Dropout(dropout_p)(x)
# Output Layer
out = Dense(num_of_classes)(x)
sm_out = Activation('softmax')(out)


#######################################
def embedding_model():
    inputs = Input(shape=(input_size,), name='input', dtype='int64')
    embedding = Embedding(vocab_size + 1,
                          embedding_size,
                          input_length=input_size,
                          weights=[embedding_weights])(inputs)
    return Model(inputs, embedding)


def conv_model():
    input_emb = Input(shape=(input_size, embedding_weights.shape[1]))
    layer1 = Conv1D(conv_layers[:1][0][0], conv_layers[:1][0][1])(input_emb)
    x = Activation('relu')(layer1)
    x = MaxPooling1D(pool_size=conv_layers[:1][0][2])(x)

    for filter_num, filter_size, pooling_size in conv_layers[1:]:
        x = Conv1D(filter_num, filter_size)(x)
        x = Activation('relu')(x)
        if pooling_size != -1:
            x = MaxPooling1D(pool_size=pooling_size)(x)
    x = Flatten()(x)  # (None, 8704)
    # Fully connected layers
    for dense_size in fully_connected_layers:
        x = Dense(dense_size, activation='relu')(x)
        x = Dropout(dropout_p)(x)
    # Output Layer
    out = Dense(num_of_classes)(x)
    return Model(input_emb, out)


def conv_keras():
    input_shape = (input_size, embedding_weights.shape[1])

    modelk = keras.models.Sequential([
        keras.layers.Conv1D(128, 7, input_shape=input_shape),
        keras.layers.Activation('relu'),
        keras.layers.MaxPool1D(3),
        keras.layers.Conv1D(128, 7),
        keras.layers.Activation('relu'),
        keras.layers.MaxPool1D(3),
        keras.layers.Conv1D(128, 3),
        keras.layers.Activation('relu'),
        keras.layers.Conv1D(64, 3),
        keras.layers.Activation('relu'),
        keras.layers.Conv1D(64, 3),
        keras.layers.Activation('relu'),
        keras.layers.Conv1D(64, 3),
        keras.layers.Activation('relu'),
        keras.layers.MaxPool1D(3),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2)
    ])
    return modelk


embd_nn = embedding_model()
conv_nn = conv_model()
sm_out = Activation('softmax')(conv_nn.output)

full_model = Model(embd_nn.input, sm_out)

exp_model = Model(conv_nn.input, conv_nn.output)

#######################################
# inputs = Input(shape=(input_size,), name='input', dtype='int64')
# emb = embedding_layer(inputs)
# conv1 = Conv1D(128, 7)(emb)
# act1 = Activation('relu')(conv1)
# maxp1 = MaxPooling1D(pool_size=3)(act1)
#
# conv2 = Conv1D(128, 7)(maxp1)
# act2 = Activation('relu')(conv2)
# maxp2 = MaxPooling1D(pool_size=3)(act2)
#
# conv3 = Conv1D(128, 3)(maxp2)
# act3 = Activation('relu')(conv3)
#
# conv4 = Conv1D(64, 3)(act3)
# act4 = Activation('relu')(conv4)
#
# conv5 = Conv1D(64, 3)(act4)
# act5 = Activation('relu')(conv5)
#
# conv6 = Conv1D(64, 3)(act5)
# act6 = Activation('relu')(conv6)
# maxp6 = MaxPooling1D(pool_size=3)(act6)
#
# fl = Flatten()(maxp6)
#
# dense1 = Dense(128, activation='relu')(fl)
# drp1 = Dropout(dropout_p)(dense1)
#
# dense2 = Dense(128, activation='relu')(drp1)
# drp2 = Dropout(dropout_p)(dense2)
#
# out = Dense(num_of_classes)(drp2)
# sm_out = Activation('softmax')(out)


#######################################
# Build model
model = Model(inputs=inputs, outputs=sm_out)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.summary()

# Shuffle
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)

x_train = train_data[indices]
y_train = train_classes[indices]

x_test = val_data
y_test = val_classes

# Training
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=32,
          epochs=12,
          verbose=2)

exp_model = Model(inputs=layer1, outputs=out)
exp_model.set_weights(model.get_weights())

# Specify methods that you would like to use to explain the model.
# Please refer to iNNvestigate's documents for available methods.
methods = ['gradient', 'lrp.z', 'lrp.alpha_2_beta_1']

# build an analyzer for each method
analyzers = []

for method in methods:
    analyzer = innvestigate.create_analyzer(method, model_without_softmax)
    analyzers.append(analyzer)

# specify indices of reviews that we want to investigate
test_sample_indices = [97, 175, 1793, 1186, 354, 1043]

test_sample_preds = [None] * len(test_sample_indices)

# a variable to store analysis results.
analysis = np.zeros([len(test_sample_indices), len(analyzers), 1, input_size])

for i, ridx in enumerate(test_sample_indices):

    x, y = x_test[ridx], y_test[ridx]

    t_start = time.time()
    x = x.reshape((1, input_size))

    pre_sm = model_without_softmax.predict_on_batch(x)[0]  # forward pass without softmax
    prob = model.predict_on_batch(x)[0]  # forward pass with softmax
    y_hat = prob.argmax()
    test_sample_preds[i] = y_hat

    for aidx, analyzer in enumerate(analyzers):
        a = np.squeeze(analyzer.analyze(x))
        a = np.sum(a, axis=1)

        analysis[i, aidx] = a
    t_elapsed = time.time() - t_start
    print('Tweet %d (%.4fs)' % (ridx, t_elapsed))

analyzer = innvestigate.create_analyzer("gradient", model)
analysis = analyzer.analyze(inputs)


def plot_text_heatmap(chars, scores, title="", width=10, height=0.2, verbose=0, max_word_per_line=20):
    fig = plt.figure(figsize=(width, height))

    ax = plt.gca()

    ax.set_title(title, loc='left')
    tokens = chars
    if verbose > 0:
        print('len chars : %d | len scores : %d' % (len(chars), len(scores)))

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)

    canvas = ax.figure.canvas
    t = ax.transData

    # normalize scores to the followings:
    # - negative scores in [0, 0.5]
    # - positive scores in (0.5, 1]
    normalized_scores = (scores / np.max(np.abs(scores)))*0.5 + 0.5

    if verbose > 1:
        print('Raw score')
        print(scores)
        print('Normalized score')
        print(normalized_scores)

    # make sure the heatmap doesn't overlap with the title
    loc_y = -0.2

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)

        text = ax.text(0.0, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'round,pad=0.5'
                       }, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()

        # create a new line if the line exceeds the length
        if (i + 1) % max_word_per_line == 0:
            loc_y = loc_y - 2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width + 15, units='dots')

    if verbose == 0:
        ax.axis('off')



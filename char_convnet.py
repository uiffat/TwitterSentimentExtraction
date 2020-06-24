import tensorflow
import keras
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from sklearn.metrics import classification_report


def load_data():
    data_df = pd.read_csv('data/train.csv')
    data_df = data_df[data_df.sentiment != 'neutral']
    split = int(0.7 * data_df.shape[0])

    train, val = np.split(data_df.sample(frac=1), [split])

    train_exp = train.selected_text
    val_exp = val.selected_text
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

    return train_data, train_classes, val_data, val_classes, train_exp, val_exp, tk


def full_model(vocab_size, input_size, word2index):
    conv_layers = [[128, 7, 3],
                   [128, 7, 3],
                   [128, 3, -1],
                   [64, 3, -1],
                   [64, 3, -1],
                   [64, 3, 3]]

    fully_connected_layers = [128, 128]
    num_of_classes = 2
    dropout_p = 0.5
    # Embedding weights
    embedding_weights = [np.zeros(vocab_size)]

    for char, i in word2index.items():
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

    model = Model(inputs=inputs, outputs=sm_out)

    return model


def embedding_model(vocab_size, input_size, word2index):
    # Embedding weights
    embedding_weights = [np.zeros(vocab_size)]

    for char, i in word2index.items():
        onehot = np.zeros(vocab_size)
        onehot[i - 1] = 1
        embedding_weights.append(onehot)

    embedding_weights = np.array(embedding_weights)
    embedding_size = embedding_weights.shape[1]
    inputs = Input(shape=(input_size,), name='input', dtype='int64')
    embedding = Embedding(vocab_size + 1,
                          embedding_size,
                          input_length=input_size,
                          weights=[embedding_weights])(inputs)
    return Model(inputs, embedding)


# defined in pure keras to make in compatible with iNNvestigate library
def conv_keras(vocab_size, input_size):
    input_shape = (input_size, vocab_size)

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


def nn_set_wts(emb_nn, conv_nn, model):
    emb_nn.layers[1].set_weights(model.layers[1].get_weights())

    for i in range(0, len(conv_nn.layers)):
        conv_nn.layers[i].set_weights(model.layers[i + 2].get_weights())

    return emb_nn, conv_nn


def decode_nn_input(x, index_word):
    sent = np.vectorize(index_word.get)(x)
    return np.array([''.join(i).split('None')[0] for i in sent])


def model_perf(model, val_X, val_Y):
    pred_label_vec = model.predict(val_X)
    pred_label = np.argmax(pred_label_vec, axis=1)
    true_label = np.argmax(val_Y, axis=1)
    target_names = ['negative', 'positive']
    print(classification_report(true_label, pred_label, target_names=target_names))


if __name__ == "__main__":
    pass

from char_convnet import *
from explanation import *
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def train_models():
    train_X, train_Y, val_X, val_Y, train_exp, val_exp, tk = load_data()

    word_index = tk.word_index
    train_exp = np.array(train_exp)
    val_exp = np.array(val_exp)

    index_word = {v: k for k, v in word_index.items()}
    input_size = 200
    vocab_size = len(word_index)

    emb_model = embedding_model(vocab_size, input_size, word_index)
    exp_model = conv_keras(vocab_size, input_size)

    char_cnn_model = full_model(vocab_size, input_size, word_index)
    char_cnn_model.summary()
    char_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    char_cnn_model.fit(train_X, train_Y,
                       validation_data=(val_X, val_Y),
                       batch_size=32,
                       epochs=12,
                       verbose=2)

    model_perf(char_cnn_model, val_X, val_Y)

    emb_model, exp_model = nn_set_wts(emb_model, exp_model, char_cnn_model)

    char_cnn_model.save('models/char_cnn_model.h5')
    emb_model.save('models/emb_model.h5')
    exp_model.save('models/exp_model.h5')

    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save('models/train_X.npy', train_X)
    np.save('models/train_Y.npy', train_Y)
    np.save('models/train_exp.npy', train_exp)
    np.save('models/val_X.npy', val_X)
    np.save('models/val_Y.npy', val_Y)
    np.save('models/val_exp.npy', val_exp)

    return True


def load_models():
    with open('models/tokenizer.pickle', 'rb') as handle:
        tk = pickle.load(handle)

    char_cnn_model = keras.models.load_model('models/char_cnn_model.h5')
    emb_model = keras.models.load_model('models/emb_model.h5')
    exp_model = keras.models.load_model('models/exp_model.h5')

    return char_cnn_model, emb_model, exp_model, tk


def run_on_tweet(tweet_text, tk, char_cnn_model, emb_model, analyser):
    word_index = tk.word_index
    index_word = {v: k for k, v in word_index.items()}

    tweet_sequence = tk.texts_to_sequences(pd.Series(tweet_text))
    tweet_data = pad_sequences(tweet_sequence, maxlen=200, padding='post')
    tweet_data = np.array(tweet_data, dtype='float32')

    pred_score = char_cnn_model.predict(tweet_data)
    pred_label = np.argmax(pred_score)
    rel_score = get_rel_score(tweet_data, emb_model, analyser)
    sent = decode_nn_input(tweet_data, index_word)
    pos_exp_text, neg_exp_text = decode_explanation(rel_score, tweet_data, index_word)

    print("\n")
    print("Tweet - ", sent)
    print("Predicted label - ", pred_label)
    print("Prediction score - ", pred_score)
    print("Generated explanations ...")
    print("Contributing to prediction - ", pos_exp_text)
    print("Driving down prediction - ", neg_exp_text)

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tweet sentiment extractor")
    parser.add_argument("--tweet", help="Tweet text")
    args = vars(parser.parse_args())
    try:
        char_cnn_model, emb_model, exp_model, tk = load_models()
    except OSError:
        train_models()
        char_cnn_model, emb_model, exp_model, tk = load_models()

    analyser = get_analyser('lrp.z', exp_model)
    run_on_tweet(args['tweet'], tk, char_cnn_model, emb_model, analyser)

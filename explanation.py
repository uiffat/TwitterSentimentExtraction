import innvestigate
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
import numpy as np
from char_convnet import decode_nn_input


def get_analyser(method, model):
    return innvestigate.create_analyzer(method, model)


def get_rel_score(x, emb_nn, analyser):
    # x = x.reshape(1, x.shape[0])
    x_emb = emb_nn.predict(x)

    analysis = analyser.analyze(x_emb)

    pool_relevance = (lambda i: np.sum(i, axis=-1))
    scores = [pool_relevance(r) for r in analysis]

    return np.array(scores)


def plot_text_heatmap(chars, scores, title="", width=10, height=0.2, verbose=0, max_char_per_line=100):
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
    normalized_scores = (scores / np.max(np.abs(scores))) * 0.5 + 0.5

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
        if (i + 1) % max_char_per_line == 0:
            loc_y = loc_y - 2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width + 15, units='dots')

    if verbose == 0:
        ax.axis('off')

    plt.show()


def decode_explanation(rel_score, x, index_word):
    sent = decode_nn_input(x, index_word)
    word_breaks = [np.concatenate([np.array([0]), np.where(i == 2)[0], np.array([len(s)])]) for i, s in zip(x, sent)]

    def get_word_rel(r, w):
        return [np.sum(r[x:y]) for x, y in zip(w[:-1], w[1:]) if x != y]

    rel_word_score = []
    for r, w in zip(rel_score, word_breaks):
        rel_word_score.append(get_word_rel(r, w))

    pad = len(max(rel_word_score, key=len))
    rel_word_score = np.array([i + [0] * (pad - len(i)) for i in rel_word_score])

    pos_rel_score = rel_word_score.copy()
    neg_rel_score = rel_word_score.copy()
    pos_rel_score[pos_rel_score < 0] = 0
    neg_rel_score[neg_rel_score > 0] = 0
    neg_rel_score = np.abs(neg_rel_score)

    ## trial log
    # pos_rel_score = np.log(pos_rel_score + 0.0001)
    # neg_rel_score = np.log(neg_rel_score + 0.0001)
    #
    # pos_rel_score = (pos_rel_score - np.min(pos_rel_score, axis=1)[:, None]) / (
    #             np.max(pos_rel_score, axis=1)[:, None] - np.min(pos_rel_score, axis=1)[:, None])
    #
    # neg_rel_score = (neg_rel_score - np.min(neg_rel_score, axis=1)[:, None]) / (
    #         np.max(neg_rel_score, axis=1)[:, None] - np.min(neg_rel_score, axis=1)[:, None])

    pos_rel_score = pos_rel_score / np.max(pos_rel_score, axis=1)[:, None]  # np.max(pos_rel_score, axis=1)[:, None]
    neg_rel_score = neg_rel_score / np.max(neg_rel_score, axis=1)[:, None]  # np.max(neg_rel_score, axis=1)[:, None]

    pos_exp_index = pos_rel_score > 0.5
    neg_exp_index = neg_rel_score > 0.5

    sent_array = np.array([s.split() + [''] * (pos_exp_index.shape[1] - len(s.split())) for s in sent])
    pos_exp = np.where(pos_exp_index == True, sent_array, '')
    neg_exp = np.where(neg_exp_index == True, sent_array, '')

    def join_exp(x):
        return ' '.join([i for i in x if i != ''])

    return np.array([join_exp(i) for i in pos_exp]), np.array([join_exp(i) for i in neg_exp])


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def analyse_explanations(char_cnn_model, emb_model, analyser, tk, val_X, val_Y, val_exp):
    word_index = tk.word_index
    index_word = {v: k for k, v in word_index.items()}

    idx = np.random.randint(0, val_X.shape[0], 5)

    x = val_X[idx]
    y = val_Y[idx]
    true_label = np.argmax(y, axis=1)
    true_exp = val_exp[idx]
    # pred_score = char_cnn_model.predict(x.reshape(1, x.shape[0]))
    pred_score = char_cnn_model.predict(x)
    pred_label = np.argmax(pred_score, axis=1)
    rel_score = get_rel_score(x, emb_model, analyser)
    sent = decode_nn_input(x, index_word)
    pos_exp_text, neg_exp_text = decode_explanation(rel_score, x, index_word)

    pos_exp_score = np.vectorize(jaccard)(true_exp, pos_exp_text)
    neg_exp_score = np.vectorize(jaccard)(true_exp, neg_exp_text)
    total_exp_score = pos_exp_score - neg_exp_score

    for i in range(len(idx)):
        print("\n")
        print("Tweet - ", sent[i])
        print("Original label - ", true_label[i])
        print("Predicted label - ", pred_label[i])
        print("Prediction score - ", pred_score[i])
        print("True explanation - ", true_exp[i])
        print("Generated explanations ...")
        print("Contributing to prediction - ", pos_exp_text[i])
        print("Driving down prediction - ", neg_exp_text[i])
        print("Total explanation quality score - ", total_exp_score[i])

    return True


if __name__ == "__main__":
    pass

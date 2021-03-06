#!/usr/bin/python3
import numpy
import pycrfsuite
import sys

"""
File to run pycrfsuite on the atis dataset, arguments are the train file, the test file, and if to use the word and char embeddings
features, embeddings are automatically imported from the data directory.
The output of this script is a file named crf.txt containing the predictions, and  model saved "model", that
can be later used with pycrsfuite.
"""

sys.path.append("..")  # to make data_manager visible from here
from data_manager import w2v_matrix_vocab_generator


def get_data(file):
    """
    Get data files (without and with feats) and combine them in data.

    :param file1: Data without feats. (i.e. NLSPARQL.train.data)
    :param file2: Data with feats. (i.e. NLSPARQL.train.feats.txt)
    :return: Data, a list of lists, each one for a sentence.
    """
    data = []
    with open(file, "r") as f:
        curr = []
        for line in f:
            if len(line) == 1:
                data.append(curr)
                curr = []
            else:
                l = line.split(" ")
                curr.append((l[0], l[-1][:-1]))
    return data


def get_w2v_embedding(word):
    """
    Return the use_embeddings vector of a word.

    :param word:
    :return:
    """
    if word in w2v_vocab:
        weights = w2v_weights[w2v_vocab[word]]
    elif word.title() in w2v_vocab:
        weights = w2v_weights[w2v_vocab[word.title()]]
    else:
        weights = numpy.zeros(300)
    return weights


def get_c2v_embedding(c):
    """
    Get the c2v embedding vector given a character.

    :param word:
    :return:
    """
    if c in w2v_vocab:
        weights = w2v_weights[w2v_vocab[c]]
    elif c.title() in w2v_vocab:
        weights = w2v_weights[w2v_vocab[c.title()]]
    else:
        weights = numpy.zeros(20)
    return weights


def w2vfeatures(sent, i):
    """
    Get use_embeddings features for a word.

    :param sent:List of lists, each word in the sentence has a list of different elements (token, pos, lemma, etc).
    :param i: Which word we are getting features for.
    :return:A dict of use_embeddings features for the word.
    """
    features = dict()
    w2v0 = get_w2v_embedding(sent[i][0])

    # current word
    for j in range(len(w2v0)):
        features["w2v_%i" % j] = w2v0[j]

    # words before the current
    if i > 0:
        w2vm1 = get_w2v_embedding(sent[i - 1][0])
        for j in range(len(w2vm1)):
            features["w2vm1_%i" % j] = w2vm1[j]
    if i > 1:
        w2vm2 = get_w2v_embedding(sent[i - 2][0])
        for j in range(len(w2vm2)):
            features["w2vm2_%i" % j] = w2vm2[j]

    if i > 2:
        w2vm3 = get_w2v_embedding(sent[i - 3][0])
        for j in range(len(w2vm3)):
            features["w2vm3_%i" % j] = w2vm3[j]

    if i > 3:
        w2vm4 = get_w2v_embedding(sent[i - 4][0])
        for j in range(len(w2vm4)):
            features["w2vm4_%i" % j] = w2vm4[j]
    if i > 4:
        w2vm5 = get_w2v_embedding(sent[i - 5][0])
        for j in range(len(w2vm5)):
            features["w2vm5_%i" % j] = w2vm5[j]

    if i > 5:
        w2vm6 = get_w2v_embedding(sent[i - 6][0])
        for j in range(len(w2vm6)):
            features["w2vm6_%i" % j] = w2vm6[j]

    # words after the current
    if i < len(sent) - 1:
        w2vp1 = get_w2v_embedding(sent[i + 1][0])
        for j in range(len(w2vp1)):
            features["w2vp1_%i" % j] = w2vp1[j]

    if i < len(sent) - 2:
        w2vp2 = get_w2v_embedding(sent[i + 2][0])
        for j in range(len(w2vp2)):
            features["w2vp2_%i" % j] = w2vp2[j]

    if i < len(sent) - 3:
        w2vp3 = get_w2v_embedding(sent[i + 3][0])
        for j in range(len(w2vp3)):
            features["w2vp3_%i" % j] = w2vp3[j]

    if i < len(sent) - 4:
        w2vp4 = get_w2v_embedding(sent[i + 4][0])
        for j in range(len(w2vp4)):
            features["w2vp4_%i" % j] = w2vp4[j]
    return features


def c2vfeatures(sent, i):
    """
    Get c2v features for a word.

    :return:A dict of use_embeddings features for the word.
    """
    features = dict()
    word = sent[i][0]
    for u, c in enumerate(word):
        c2v0 = get_c2v_embedding(c)
        for j in range(len(c2v0)):
            features["c2v_%i_%i" % (u, j)] = c2v0[j]
    return features


def other_features(sent, i):
    """
    Get token features (not related to embeddings) for a word.

    :param sent:List of lists, each word in the sentence has a list of different elements (token, pos, lemma, etc).
    :param i: Which word we are getting features for.
    :return:A dict of use_embeddings features for the word.
    """
    features = dict()
    word = sent[i][0]

    features['word'] = word
    features['word[-3]'] = word[-3:]
    features['word[3]'] = word[:3]

    # words before the current
    if i > 0:
        wordm1 = sent[i - 1][0]
        prem1 = sent[i - 1][0][:3]
        features["-1-word"] = wordm1
        features["-1-pre"] = prem1
        features["-1-conj"] = wordm1 + word
    else:
        features['BOS'] = 1

    if i > 1:
        wordm2 = sent[i - 2][0]

        features["-2-word"] = wordm2

    if i > 2:
        wordm3 = sent[i - 3][0]

        features["-3-word"] = wordm3

    if i > 3:
        wordm4 = sent[i - 4][0]

        features["-4-word"] = wordm4

    if i > 4:
        wordm5 = sent[i - 5][0]

        features["-5-word"] = wordm5

    if i > 5:
        wordm6 = sent[i - 6][0]

        features["-6-word"] = wordm6

    # words after the current
    if i < len(sent) - 1:
        wordp1 = sent[i + 1][0]

        features["1-word"] = wordp1
        features["1-conj"] = word + wordp1
    else:
        features['EOS'] = 1

    if i < len(sent) - 2:
        wordp2 = sent[i + 2][0]

        features["2-word"] = wordp2

    if i < len(sent) - 3:
        wordp3 = sent[i + 3][0]

        features["3-word"] = wordp3

    if i < len(sent) - 4:
        wordp4 = sent[i + 4][0]

        features["4-word"] = wordp4

    return features


def word2features(sent, i, w2v=True, c2v=True, other=True):
    """
    Get features for a word.

    :param sent:List of lists, each word in the sentence has a list of different elements (token, pos, lemma, etc).
    :param i: Which word we are getting character features for.
    :param w2v: True if use_embeddings features are to be added.
    :param c2v: True if c2v features are to be added.
    :param other: True if other (non embedding related) features are to be added.
    :return:
    """
    features = {}
    if w2v:
        features = {**w2vfeatures(sent, i), **features}
    if c2v:
        features = {**c2vfeatures(sent, i), **features}
    if other:
        features = {**other_features(sent, i), **features}

    return features


def sent2features(sent, w2v):
    return [word2features(sent, i, w2v, w2v, True) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        print("usage is:\n ./atis_run.py train_file test_file, or ./movies_run.py train_file test_file w2v_pickle "
              "c2v_pickle w2v embeddings and char embeddings")
        print("Train and test files are in 1 word per line format, w2v and c2v are pickles mapping a word or a "
              "character to its google embedding vector or char embedding.")
        exit()
    train, test = sys.argv[1], sys.argv[2]
    use_embeddings = len(sys.argv) == 5
    if use_embeddings:
        w2v_vocab, w2v_weights = w2v_matrix_vocab_generator(sys.argv[3])
        c2v_vocab, c2v_weights = w2v_matrix_vocab_generator(sys.argv[4])

    train = get_data(train)
    test = get_data(test)

    trainer = pycrfsuite.Trainer(verbose=True)
    # TRAINING
    X_train = ([pycrfsuite.ItemSequence(sent2features(s, use_embeddings)) for s in train])

    y_train = [sent2labels(s) for s in train]

    X_test = ([pycrfsuite.ItemSequence(sent2features(s, use_embeddings)) for s in test])
    y_test = [sent2labels(s) for s in test]

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 0.0,  # coefficient for L1 penalty
        'c2': 0.10,  # coefficient for L2 penalty 0.2
        'num_memories': 100,
        'max_iterations': 200,  # stop earlier
        'max_linesearch': 20,
        'feature.possible_states': 1,
        'feature.possible_transitions': 1,
        'feature.minfreq': 0.000000,
    })

    print("training model, saving as 'model'")
    trainer.train("model")
    print("done")

    # TESTING
    tagger = pycrfsuite.Tagger()
    tagger.open("model")

    print("saving test data as crf.txt")
    with open("crf.txt", "w") as output:
        for sent in test:
            predicted = tagger.tag(sent2features(sent, use_embeddings))
            correct = sent2labels(sent)
            tokens = sent2tokens(sent)
            for t, p, c in zip(tokens, predicted, correct):
                output.write("%s %s %s\n" % (t, p, c))
            output.write("\n")

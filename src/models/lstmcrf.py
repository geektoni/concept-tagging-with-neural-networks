import torch
import torch.nn as nn
from torch.autograd import Variable

import data_manager


# recurrent-crf implementation, heavily inspired by the pytorch tutorial and by kaniblu, the CRF class is almost
# untouched, while the lstm-crf class has substantial changes, among which is the convolution on char embeddings.

def sequence_mask(lengths, max_len, device):
    batch_size = lengths.size(0)

    ranges = torch.arange(0, max_len).long().to(device)
    ranges = ranges.expand(batch_size, -1)
    lens_exp = lengths.unsqueeze(1).expand_as(ranges)
    # set 1 where [batch][i] with i < length of phrase
    mask = ranges < lens_exp
    return mask


class CRF(nn.Module):
    def __init__(self, device, vocab_size):
        super(CRF, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.n_labels = n_labels = vocab_size + 2
        self.tagset_size = self.n_labels
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels), requires_grad=True)

    @staticmethod
    def log_sum_exp(vec, dim=0):
        """
        Numerically stable log sum exp.
        :param vec: Vector of values.
        :param dim: Dimension over which the log sum exp is being done.
        :return: Log sum exp value/s.
        """
        max, idx = torch.max(vec, dim)
        max_exp = max.unsqueeze(-1).expand_as(vec)
        return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

    def forward(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, tagset_size] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        alpha = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            # expand tag scores over columns
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            # expand score of tags at previous time step over rows
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            # expand transitions over batchs
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)

            # obtain scores of tags for this step
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = self.log_sum_exp(mat, 2).squeeze(-1)

            # update alpha, get alpha of current step + carry over alphas of sentences that have already been finished
            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1

        # last step
        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = self.log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, tagset_size] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        vit[:, self.start_idx] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[self.stop_idx].unsqueeze(0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(list(reversed(pointers)))
        scores, idx = vit.max(1, keepdim=True)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        for argmax in pointers:
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)
            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self, labels, lengths):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lengths: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = labels.data.new(batch_size, seq_len + 2)
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels

        mask = sequence_mask(lengths + 1, seq_len + 2, self.device).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.stop_idx))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lengths + 1, lengths.max() + 1, self.device).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score


class LstmCrf(nn.Module):
    def __init__(self, device, w2v_weights, tag_to_itx, hidden_dim, drop_rate, bidirectional=False, freeze=True,
                 embedding_norm=6, c2v_weights=None, pad_word_length=16, embedder="none", more_features=False):

        super(LstmCrf, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag_to_itx)
        self.embedding_dim = w2v_weights.shape[1]
        self.w2v_weights = w2v_weights
        self.c2v_weights = c2v_weights
        self.pad_word_length = pad_word_length
        self.bidirectional = bidirectional
        self.embedder = embedder
        self.more_features = more_features

        self.drop_rate = drop_rate
        self.drop = nn.Dropout(self.drop_rate)

        # embedding layer
        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=freeze)
        self.embeddings.max_norm = embedding_norm

        # Use the Elmo embedder instead of the classical ones.
        if self.embedder != "none":
            self.embeddings = None
            self.embedding_dim = 768 if self.embedder == "bert" else 1024

        # We add the dimensionality of the other features (POS and spaCy).
        if self.more_features:
            self.embedding_dim += 58 + 18

        # recurrent and mapping to tagset
        self.recurrent = nn.LSTM(input_size=self.embedding_dim,
                                 hidden_size=self.hidden_dim // (1 if not self.bidirectional else 2),
                                 bidirectional=self.bidirectional, batch_first=True)
        self.bnorm = nn.BatchNorm2d(1)
        self.fc = nn.Linear(self.hidden_dim, self.tagset_size + 2)  # + 2 because of start and end token
        self.bnorm2 = nn.BatchNorm2d(1)

        # crf for scoring at a global level
        self.crf = CRF(self.device, self.tagset_size)

        # setup convolution on characters if c2v_weights are passed
        if self.c2v_weights is not None:
            self.char_embedding_dim = c2v_weights.shape[1]
            self.char_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(c2v_weights), freeze=True)
            self.char_embedding.max_norm = embedding_norm
            self.feats = 20  # for the output channels of the conv layers

            self.recurrent = nn.LSTM(self.embedding_dim + 50,
                                     self.hidden_dim // (1 if not self.bidirectional else 2),
                                     batch_first=True, bidirectional=self.bidirectional)

            # conv layers for single character, pairs of characters, 3x characters
            self.ngram1 = nn.Sequential(
                nn.Conv2d(1, self.feats * 1, kernel_size=(1, self.char_embedding_dim),
                          stride=(1, self.char_embedding_dim),
                          padding=0),
                nn.Dropout2d(p=self.drop_rate),
                nn.MaxPool2d(kernel_size=(self.pad_word_length, 1)),
                nn.Tanh(),
            )

            self.ngram2 = nn.Sequential(
                nn.Conv2d(1, self.feats * 2, kernel_size=(2, self.char_embedding_dim),
                          stride=(1, self.char_embedding_dim),
                          padding=0),
                nn.Dropout2d(p=self.drop_rate),
                nn.MaxPool2d(kernel_size=(self.pad_word_length - 1, 1)),
                nn.Tanh(),
            )

            self.ngram3 = nn.Sequential(
                nn.Conv2d(1, self.feats * 3, kernel_size=(3, self.char_embedding_dim),
                          stride=(1, self.char_embedding_dim),
                          padding=0),
                nn.Dropout2d(p=self.drop_rate),
                nn.MaxPool2d(kernel_size=(self.pad_word_length - 2, 1)),
                nn.Tanh(),
            )

            # seq layers to elaborate on the output of conv layers
            self.fc1 = nn.Sequential(
                nn.Linear(self.feats, 10),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(self.feats * 2, 20),
            )
            self.fc3 = nn.Sequential(
                nn.Linear(self.feats * 3, 20),
            )

    def features_score(self, feats, labels, lengths):
        """
        Given the label scores (feats) of each token and the correct labels,
        return the score of the whole sentence.
        :param feats: Label scores for each token, size = (batch, sentence length, tagset size)
        :param labels: Correct label of each word, size = (batch, sentence length)
        :param lengths: Lengths of each sentence, needed for masking out padding. size = (batch)
        :return: Score of each sentence, size = (batch)
        """
        batch_size, max_length = labels.size()
        labels_exp = labels.unsqueeze(-1)
        # set paddings to 0
        labels[labels == -1] = 0
        # get the score that was given to each correct label
        scores = torch.gather(feats, 2, labels_exp).squeeze(-1)
        # mask out scores of padding
        mask = sequence_mask(lengths, max_length, self.device).float()
        scores = scores * mask

        # sum and return
        score = scores.sum(1).squeeze(-1)
        return score

    @staticmethod
    def get_labels(labels, padding=-1):
        """
        Get labels of each sentence, keeping only as much as needed (up to the length of the longest sentence).
        :param labels: Labels of each word for each sentence.
        :param padding: Padding value to use, default -1.
        :return: Labels of each sentence, size = (batch, longest sentence).
        """
        tmp = labels != padding
        tmp = torch.sum(tmp, dim=1)
        max_length = torch.max(tmp)
        res = labels[:, :max_length]
        return res

    def init_hidden(self, batch_size):
        """
        Inits the hidden state of the recurrent layer.
        :param batch_size
        :return: Initialized hidden state of the recurrent layer.
        """
        if self.bidirectional:
            state = [torch.zeros(self.recurrent.num_layers * 2, batch_size, self.hidden_dim // 2).to(self.device),
                     torch.zeros(self.recurrent.num_layers * 2, batch_size, self.hidden_dim // 2).to(self.device)]
        else:
            state = [torch.zeros(self.recurrent.num_layers, batch_size, self.hidden_dim).to(self.device),
                     torch.zeros(self.recurrent.num_layers, batch_size, self.hidden_dim).to(self.device)]
        return state

    def get_features_from_recurrent(self, data, char_data, lengths):
        """
        For each word get its scores for each possible label.
        :param data: Input sentences.
        :param lengths: Lengths of each sentence, needed for packing.
        :return: Labels scores of each token, size = (batch, sentence length, tagset size)
        """
        # n_feats, batch_size, seq_len = xs.size()
        if self.embedder != "none":
            batch_size, seq_len, enc = data.size()
        else:
            batch_size, seq_len = data.size()

        # embed and drop
        if self.embeddings is not None:
            embedded = self.embeddings(data)
        else:
            embedded = data

        embedded = embedded.view(batch_size, seq_len, self.embedding_dim)
        embedded = self.drop(embedded)

        if self.c2v_weights is not None:
            batched_conv = []
            char_data = self.char_embedding(char_data)
            char_data = self.drop(char_data)
            num_words = char_data.size()[2]
            for i in range(num_words):
                # get word for each batch, then convolute on the ith word of each batch and concatenate
                c = char_data[:, 0, i, :, :].unsqueeze(1)
                ngram1 = self.ngram1(c).view(char_data.size()[0], 1, 1, -1)
                ngram2 = self.ngram2(c).view(char_data.size()[0], 1, 1, -1)
                ngram3 = self.ngram3(c).view(char_data.size()[0], 1, 1, -1)
                ngram1 = self.fc1(ngram1)
                ngram2 = self.fc2(ngram2)
                ngram3 = self.fc3(ngram3)
                batched_conv.append(torch.cat([ngram1, ngram2, ngram3], dim=3))
            batched_conv = torch.cat(batched_conv, dim=1).squeeze(2)
            embedded = torch.cat([embedded, batched_conv], dim=2)

        # pack, pass through recurrent, unpack
        packed = nn.utils.rnn.pack_padded_sequence(embedded, sorted(lengths.data.tolist(), reverse=True),
                                                   batch_first=True)
        hidden = self.init_hidden(batch_size)
        output, _ = self.recurrent(packed, hidden)
        o, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # pass through fc layer and activation
        o = o.contiguous()
        o = self.bnorm(o.unsqueeze(1)).squeeze(1)
        o = self.fc(o)
        o = self.bnorm2(o.unsqueeze(1)).squeeze(1)
        return o

    def score(self, feats, labels, lengths):
        """
        Score a sentence given scores from the recurrent layer and transition scores from the crf.
        :param feats: Label scores for each token, size = (batch, sentence length, tagset size)
        :param labels: Correct label of each word, size = (batch, sentence length)
        :param lengths: Lengths of each sentence, needed for masking out padding. size = (batch)
        :return: Sentence score, size = (batch)
        """
        transition_score = self.crf.transition_score(labels, lengths)
        bilstm_score = self.features_score(feats, labels, lengths)

        score = transition_score + bilstm_score
        return score

    def forward(self, batch):
        data, labels, char_data, pos, ner = data_manager.batch_sequence(batch, self.device)
        lengths = self.get_lengths(labels)

        # If we are using more features the we concatenate everything together
        if self.more_features:
            data = torch.cat([data, pos, ner], 2)

        # get features and do predictions maximizing the sentence score using the crf
        feats = self.get_features_from_recurrent(data, char_data, lengths)
        scores, predictions = self.crf.viterbi_decode(feats, lengths)

        # pad predictions so that they match in length with padded labels
        batch_size, pad_to = labels.size()
        _, pad_from = predictions.size()
        padding = torch.zeros(batch_size, pad_to - pad_from).long().to(self.device)
        predictions = torch.cat([predictions, padding], dim=1)
        predictions = predictions.expand(*labels.size())

        # remove start and stop tags if there are any (mostly for safety, should not happen)
        predictions[predictions == 43] = 0
        predictions[predictions == 44] = 0

        return predictions.view(-1), labels.view(-1)

    def get_lengths(self, labels, padding=-1):
        """
        Get length of each sentences.
        :param labels: Labels of each word for each sentence.
        :param padding: Padding value to use, default -1.
        :return: Length of each sentence, size = (batch).
        TODO: remove for cycle, make it with matrix operations
        """
        batchs, _ = labels.size()
        lengths = torch.zeros(batchs).long().to(self.device)
        for i in range(batchs):
            while len(labels[i]) > lengths[i] and labels[i][lengths[i]] != padding:
                lengths[i] += 1
        return lengths

    def neg_log_likelihood(self, batch):
        """
        Used for training, returns a loss that depends on the difference between the score that the model
        would give to the sentence annd the score it would give to the correct labeling of the sentence.
        :param batch:
        :return:Pytorch loss.
        """
        data, labels, char_data, pos, ner = data_manager.batch_sequence(batch, self.device)
        lengths = self.get_lengths(labels)
        labels = self.get_labels(labels)

        # If we are using more features the we concatenate everything together
        if self.more_features:
            data = torch.cat([data, pos, ner], 2)

        # get feats (scores for each label, for each word) from recurrent
        feats = self.get_features_from_recurrent(data, char_data, lengths)
        # get score of sentence from crf
        norm_score = self.crf(feats, lengths)

        # get score that the model would give to the correct labels
        sequence_score = self.score(feats, labels, lengths)

        loglik = sequence_score - norm_score
        loglik = -loglik.mean()
        return loglik


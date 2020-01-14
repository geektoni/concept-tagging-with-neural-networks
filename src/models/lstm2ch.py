import torch
import torch.nn as nn
import torch.nn.functional as F

import data_manager


class LSTM2CH(nn.Module):
    def __init__(self, device, w2v_weights, hidden_dim, tagset_size, drop_rate, bidirectional=False,
                 embedding_norm=10., embedder="none", more_features=False):
        """
        :param device: Device to which to map tensors (GPU or CPU).
        :param w2v_weights: Matrix of w2v w2v_weights, ith row contains the embedding for the word mapped to the ith index, the
        last row should correspond to the padding token, <padding>.
        :param hidden_dim Size of the hidden dimension of the recurrent layer.
        :param tagset_size: Number of possible classes, this will be the dimension of the output vector.
        :param drop_rate: Drop rate for regularization.
        :param bidirectional: If the recurrent should be bidirectional.
        :param embedding_norm: Max norm of the dynamic embeddings.
        """
        super(LSTM2CH, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.embedding_dim = w2v_weights.shape[1]
        self.w2v_weights = w2v_weights
        self.bidirectional = bidirectional
        self.embedder = embedder
        self.more_features = more_features

        self.embedding_static = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=True)
        self.embedding_dyn = nn.Embedding(w2v_weights.shape[0], w2v_weights.shape[1], max_norm=embedding_norm,
                                          scale_grad_by_freq=True)

        recurrent_dyn_dim = w2v_weights.shape[1]

        # Use the Elmo embedder instead of the classical ones.
        if self.embedder != "none":
            self.embedding_static = None
            self.embedding_dim = 768 if self.embedder == "bert" else 1024


        # We add the dimensionality of the other features (POS and spaCy).
        if self.more_features:
            self.embedding_dim += 58 + 18
            recurrent_dyn_dim += 58 + 18

        # Create new dynamic embedding with the new size
        #if self.more_features or self.embedder != "none":
        #    self.embedding_dyn = nn.Embedding(w2v_weights.shape[0], self.embedding_dim, max_norm=embedding_norm,
        #                                      scale_grad_by_freq=True)
        self.drop_rate = drop_rate
        self.drop = nn.Dropout(self.drop_rate)

        # two "parallel" recurrent layers, 1 for static and 1 for dynamic embeddings
        self.recurrent_static = nn.LSTM(self.embedding_dim, self.hidden_dim // (2 if not bidirectional else 4),
                                        batch_first=True, bidirectional=bidirectional)
        self.recurrent_dyn = nn.LSTM(recurrent_dyn_dim, self.hidden_dim // (2 if not bidirectional else 4),
                                     batch_first=True, bidirectional=bidirectional)

        self.hidden2tag = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_dim, self.tagset_size),
            nn.ReLU(inplace=True)
        )

    def init_hidden(self, batch_size):
        """
        Inits the hidden state of the recurrent layer.
        :param batch_size
        :return: Initialized hidden state of the recurrent encoder.
        """
        if self.bidirectional:
            state = [
                torch.zeros(self.recurrent_static.num_layers * 2, batch_size, self.hidden_dim // 4).to(self.device),
                torch.zeros(self.recurrent_static.num_layers * 2, batch_size, self.hidden_dim // 4).to(self.device)]
        else:
            state = [torch.zeros(self.recurrent_static.num_layers, batch_size, self.hidden_dim // 2).to(self.device),
                     torch.zeros(self.recurrent_static.num_layers, batch_size, self.hidden_dim // 2).to(self.device)]
        return state

    def forward(self, batch):
        """
        Forward pass given data.
        :param batch: List of samples containing data as transformed by the init transformer of this class.
        :return: A (batch of) vectors of length equal to tagset, scoring each possible class for each word in a sentence,
        for all sentences; a tensor containing the true label for each word and a tensor containing the lengths
        of the sequences in descending order.
        """
        hidden_static = self.init_hidden(len(batch))
        hidden_dyn = self.init_hidden(len(batch))

        # embed using static embeddings and pass through the recurrent layer
        data, labels, char_data, pos, ner, data_index = data_manager.batch_sequence(batch, self.device, True)

        if self.embedding_static is not None:
            data_static = self.embedding_static(data)
        else:
            data_static = data

        # add the new features
        if self.more_features:
            data_static = torch.cat([data_static, pos, ner], 2)

        data_static = self.drop(data_static)
        lstm_out_static, hidden_static = self.recurrent_static(data_static, hidden_static)

        # embed using dynamic embeddings and pass through the recurrent layer
        data_dynamic = self.embedding_dyn(data_index)
        # add the new features
        if self.more_features:
            data_dynamic = torch.cat([data_dynamic, pos, ner], 2)
        data_dynamic = self.drop(data_dynamic)
        lstm_out_dyn, hidden_dyn = self.recurrent_dyn(data_dynamic, hidden_dyn)

        # concatenate results
        output = torch.cat([lstm_out_static, lstm_out_dyn], dim=2)

        # send output to fc layer(s)
        tag_space = self.hidden2tag(output.unsqueeze(1).contiguous())
        tag_scores = F.log_softmax(tag_space, dim=3)

        return tag_scores.view(-1, self.tagset_size), labels.view(-1)

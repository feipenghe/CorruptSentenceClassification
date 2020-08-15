import torch
import torch.nn as nn
from docutils.nodes import target
import numpy as np

def iterable2tensor(tuple):
    '''
    Transform tensors in tuple into a single tensor
    Example input: (tensor([1]), tensor([0]))   |   [tensor([1]), tensor([0])
    Example output: tensor([[1],[0]])
    :param tuple:
    :return: tensor
    '''
    tensor = None
    for t in tuple:
        if tensor == None:
            tensor = torch.unsqueeze(t, dim=0)
        else:
            t = torch.unsqueeze(t, dim=0)
            tensor = torch.cat((tensor, t), dim=0)
    return tensor

def collate_fn(batch):
    """
    Create a batch of data given a list of N sequences and labels. Sequences are stacked into a single tensor
    of shape (N, max_sequence_length), where max_sequence_length is the maximum length of any sequence in the
    batch. Sequences shorter than this length should be filled up with 0's. Also returns a tensor of shape (N, 1)
    containing the label of each sequence.

    :param batch: A list of size N, where each element is a tuple containing a sequence tensor and a single item
    tensor containing the true label of the sequence.

    :return: A tuple containing two tensors. The first tensor has shape (N, max_sequence_length) and contains all
    sequences. Sequences shorter than max_sequence_length are padded with 0s at the end. The second tensor
    has shape (N, 1) and contains all labels.
    """
    sentences, labels = zip(*batch)

    # Determine the maximum number of sequence length
    max_sequence_length = -1
    for sent in sentences:
        temp_len = sent.shape[0]
        if temp_len > max_sequence_length:
            max_sequence_length = temp_len
    pad_t = torch.tensor([0]).long()
    padded_sents = []

    for i in range(len(sentences)): # pad each sent
        cur_sent_t = sentences[i]
        num_pad = max_sequence_length - cur_sent_t.shape[0]
        padded_t = torch.cat( (cur_sent_t.long(), pad_t.repeat(num_pad).long() ) )
        padded_sents.append(padded_t)



    padded_sents = iterable2tensor(padded_sents)
    labels = iterable2tensor(labels)
    print(padded_sents)
    print(labels)
    exit()
    return padded_sents, labels


class RNNBinaryClassificationModel(nn.Module):
    def __init__(self, embedding_matrix, device ):
        super().__init__()

        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        # Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.embedding.weight.data = embedding_matrix

        self.output_size = 1
        self.hidden_dim = 200
        self.n_layers = 1
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first = True)
        self.fc = nn.Linear(self.hidden_dim * self.n_layers, self.output_size)
        self.criterion = nn.BCELoss()
        # self.criterion = nn.HingeEmbeddingLoss()
        self.sm = nn.Sigmoid()

        self.device = device

    def forward(self, inputs):
        """
        Takes in a batch of data of shape (N, max_sequence_length). Returns a tensor of shape (N, 1), where each
        element corresponds to the prediction for the corresponding sequence.
        :param inputs: Tensor of shape (N, max_sequence_length) containing N sequences to make predictions for.
        :return: Tensor of predictions for each sequence of shape (N, 1).
        """

        # inputs -> embedding
        embedding_input = self.embedding(inputs)  # inputs shape  2 x 30

        batch_size = embedding_input.size(0)


        h0= torch.zeros(self.n_layers , inputs.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.n_layers , inputs.size(0), self.hidden_dim).requires_grad_().to(self.device)


        # out, hidden= self.rnn(embedding_input)  # GRU or GPU
        try:
            pred_out, (hidden, cell_state) = self.rnn(embedding_input, (h0.detach(), c0.detach())) # output 2 x max_seq_length x 64    # LSTM
        except RuntimeError as re:
            print("inputs: ", inputs)
            print("embedding: ", embedding_input)
            print(re)
        # hidden = torch.squeeze(hidden)
        # out = self.fc(out[:, -1, :])
        # print("hidden dim: ", hidden.shape)
        # hidden = hidden.permute(1, 0, 2)
        # out = self.fc(torch.reshape(hidden, (hidden.shape[1],-1)))
        # out = torch.squeeze(hidden)
        # print("pred out: ", pred_out.shape)
        # print(self.embedding("they"))
        # exit()
        out= hidden.squeeze()
        # print("out: ", out)

        out = self.fc(out)
        # print("out1: ", out)
        out = self.sm(out)
        # print("out2: ", out)

        return out

    def loss(self, logits, targets):
        """
        Computes the binary cross-entropy loss.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Binary cross entropy loss between logits and targets as a scalar tensor.
        """
        return self.criterion(logits, targets.float())


    def accuracy(self, logits, targets):
        """
        Computes the accuracy, i.e number of correct predictions / N.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Accuracy as a scalar tensor.
        """
        num_correct = 0
        # targets = targets.float()
        for i in range(len(logits)):
            pred = torch.round(logits[i])
            targets = targets.float()
            # print("pred: ", pred.dtype)
            # print(pred)
            # print("target: ", targets[i].dtype)
            # print(targets[i])
            # exit()
            if pred == targets[i]:
                num_correct += 1
        return torch.tensor(num_correct*1.0/len(logits))


# Training parameters
TRAINING_BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4 # 0.001 acc went down
# LEARNING_RATE = 0.0001
# Batch size for validation, this only affects performance.
VAL_BATCH_SIZE = 128

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        # linear layer has two parameters: weights and biases
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        # initialize the parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.bias)

    def forward(self, inputs):
        """Compute f(x) = xW + b.

        Arguments:
            inputs: a tensor of shape (batch_size, input_size)

        Returns:
            a tensor of shape (batch_size, output_size)
        """
        # TODO: implement the forward pass of a linear layer.
        # You will want to use torch.matmul, as well as tensor addition
        return torch.matmul(inputs, self.weight) + self.bias


class DeepAveragingNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(DeepAveragingNetwork, self).__init__()

        # embedding matrix
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=1)
        # first hidden layer
        self.hidden = LinearLayer(embedding_dim, hidden_dim)
        # second hidden layer
        self.hidden2 = LinearLayer(hidden_dim, hidden_dim)
        # output layer
        self.output = LinearLayer(hidden_dim, num_classes)

    def forward(self, text, padding_index):
        # (batch_size, seq_len, embed_dim)
        embeddings = self.embedding(text)  # embeddings for the batch
        # To take the average of embeddings, we need to know the length of each
        # input (each row of text).  Padding_index tells us which token ID
        # corresponds to padded positions.  So, to get length, we code all
        # positions with real tokens as a 1, all positions with a padding token
        # as a 0, and then sum along the rows.
        # (batch_size)
        lengths = (text != padding_index).sum(dim=1)
        # (batch_size, 1)
        lengths = torch.unsqueeze(lengths, 1) # text length for each batch while maintaining their tensor data type
        # (batch_size, embed_dim)
        bagged = torch.sum(embeddings, dim=1) / lengths
        # (batch_size, hidden_dim)
        hidden = F.relu(self.hidden(bagged))
        hidden = F.relu(self.hidden2(hidden))
        # (batch_size, num_classes)
        logits = self.output(hidden)
        return logits

import torch
import torch.nn as nn


class CorruptionGenerator(nn.Module):

    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim = embedding_dim, padding_idx=0)
        self.embedding.weight.data = embedding_matrix

        self.output_size = embedding_dim # output a token embedding
        self.hidden_dim = 100
        self.n_layers = 1
        self.bidirect = False
        if self.bidirect:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True)

        self.criterion = nn.CrossEntropyLoss()
        # this linear layer is decoder
        self.fc = nn.Linear(self.hidden_dim * self.n_layers , vocab_size)
        self.vocab_size = vocab_size
        # self.device = device
    def forward(self, inputs, prev_state):
        embedding_input = self.embedding(inputs)
        # TODO: move the initial states outside
        # print("prev state: ", prev_state[0].shape)
        # print("embedding_input shape: ", embedding_input.shape)
        output, state = self.rnn(embedding_input, prev_state )
        # print("rnn output shape: ", output.shape)
        logits = self.fc(output.reshape(output.size(0)*output.size(1),output.size(2) ))
        # print("logits shape: ", logits.shape)
        logits = logits.reshape(output.size(0), output.size(1), logits.size(1))
        # print("logitis after reshaping: ", logits.shape)
        return logits, state

    def init_state(self, batch_size):
        '''
        Initialize states
        :param sequence_length: can be obtained by input.size(0)
        :return: hidden state and cell state
        '''
        h0 = torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_dim).requires_grad_().cuda()
        return (h0, c0)

    def loss(self, logits, targets):
        # print("logits shape: ", logits.view(-1, ).shape)
        return self.criterion(logits.view(-1, self.vocab_size), targets)
    def accuracy(self, logits, targets):
        # print("accuracy logits: ", logits.shape)
        # print("target: ", targets.shape)
        preds =torch.argmax(logits,dim = 2)
        preds = preds.view(-1)
        num_correct = torch.sum( (preds == targets).int())
        #
        # for i in range(len(logits)):
        #     pred = torch.round(logits[i])
        #     targets = targets
        #     # print("pred: ", pred.dtype)
        #     print(pred)
        #
        #     # print("target: ", targets[i].dtype)
        #     # print(targets[i])
        #     # exit()
        #     print(pred.shape)
        #     print("prediction: ", )
        #     exit()
        #     if torch.argmax(pred, dim=1) == targets[i]:
        #         num_correct += 1
        # if (num_correct > preds.size(0)):
        # print("prediction size: ", (preds.size(0)))
        #
        # print("num correct: ", num_correct)

            # exit()
        return torch.tensor(num_correct.item() * 1.0 /(preds.size(0)))
class CorruptionDiscriminator(nn.Module):
    def __init__(self, embedding_matrix, device ):
        super().__init__()

        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        # Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.embedding.weight.data = embedding_matrix

        self.output_size = 1
        self.hidden_dim = 100
        self.n_layers = 3
        self.bidirect = True
        if self.bidirect :
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first = True, bidirectional= self.bidirect )


        self.fc1 = nn.Linear(self.hidden_dim * self.n_layers , self.hidden_dim * self.n_layers) # consider giving up this structure
        self.fc2 = nn.Linear(self.hidden_dim * self.n_layers * self.num_directions,self.n_layers * self.num_directions)
        self.fc3 = nn.Linear(self.n_layers * self.num_directions, 1)
        self.sm = nn.Sigmoid()
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.HingeEmbeddingLoss()
        self.criterion = nn.BCELoss()


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

        # h0= torch.zeros(self.n_layers * self.num_directions, inputs.size(0), self.hidden_dim).requires_grad_().cuda()
        # c0 = torch.zeros(self.n_layers * self.num_directions, inputs.size(0), self.hidden_dim).requires_grad_().cuda()

        try:
            _, hidden = self.rnn(embedding_input)  # GRU OR RNN
            # pred_out, (hidden, cell_state) = self.rnn(embedding_input, (h0.detach(), c0.detach())) # output 2 x max_seq_length x 64    # LSTM
        except RuntimeError as re:
            print("inputs: ", inputs)
            print("embedding: ", embedding_input)
            print(re)
            exit()

        out= hidden.squeeze()
        # print("out: ", out.shape)

        out = out.transpose(0, 1).contiguous() # batch_size x #layers x hidden_dim
        out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        # print("out1: ", out.shape)
        # exit()
        # print("out2: ", out.shape)
        out = self.fc2(out)

        out = self.fc3(out)
        out = self.sm(out)

        # out = self.fc3(out)
        # print("out: ", out)

        # exit()

        return out

    def loss(self, logits, targets):
        """ tensor([0.5774],
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
TRAINING_BATCH_SIZE = 300
NUM_EPOCHS = 25
LEARNING_RATE = 5e-5 # 0.001 acc went down
# LEARNING_RATE = 0.0001
# Batch size for validation, this only affects performance.
VAL_BATCH_SIZE = 500
TEST_BATCH_SIZE = 10

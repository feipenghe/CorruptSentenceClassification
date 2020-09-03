import torch
from torch.utils.data import Dataset

import csv

from collections import Counter

def iterable2tensor(iterable):
    '''
    Transform tensors in tuple into a single tensor
    Example input: (tensor([1]), tensor([0]))   |   [tensor([1]), tensor([0])
    Example output: tensor([[1],[0]])
    :param iterable:
    :return: tensor
    '''
    tensor = None
    for t in iterable:
        if tensor == None:
            tensor = torch.unsqueeze(t, dim=0)
            # print(tensor.shape)
        else:
            t = torch.unsqueeze(t, dim=0)
            tensor = torch.cat((tensor, t), dim=0)
    return tensor


def iterable2tensor2(iterable):
    '''
    Transform tensors in tuple into a single tensor
    Example input: (tensor([1]), tensor([0]))   |   [tensor([1]), tensor([0])
    Example output: tensor([[1],[0]])
    :param iterable:
    :return: tensor
    '''
    tensor = None
    # print(len(iterable))
    # print(iterable[0].shape)
    # exit()
    for t in iterable:
        if tensor == None:
            tensor = t
        else:
            # t = torch.unsqueeze(t, dim=0)
            try:
                tensor = torch.cat((tensor, t), dim=0)
            except RuntimeError as e:
                print(e)
                print(tensor)
                print(t)
                exit()

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
    return padded_sents, labels


def seq_collate_fn(batch):
    """ Collate one batch of sentences
    :param one batch of sentence data
    :return sequence inputs with size (seq_batch_size, seq_length), labels with size (num_seq*bptt, 1) and num_seq
    """
    bptt = 3


    # pad sentences
    sentences = batch
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



    # build sequences
    def get_seq_batch(sent_batch, i, bptt):
        seq_len = min(bptt, sent_batch.size(1) - 1 - i)
        # TODO: check if this indexing is correct
        seq_data = sent_batch[:, i:i + seq_len]
        seq_target = sent_batch[:, i + 1:i + 1 + seq_len]
        return seq_data, seq_target


    seq_inputs = []
    seq_targets = []
    for seq_batch_idx, i in enumerate(range(0, max_sequence_length, bptt)):
        input, target = get_seq_batch(padded_sents, i, bptt)
        # TODO: might affect    the accuray of predicting sentence ends
        if input.size(1) == bptt:
            seq_inputs.append(input)
            seq_targets.append(target)
    # TODO: input tensor has a special i2t method
    seq_inputs = iterable2tensor2(seq_inputs)
    # print("label before iterable: ", len(seq_targets) , "   ", seq_targets[0].shape)
    labels = iterable2tensor2(seq_targets)
    # seq_inputs = seq_inputs.transpose(1,0)
    # print("input shape: ", seq_inputs.shape)
    # print("label shape: ",  labels.view(-1).shape)
    # exit()
    return seq_inputs, labels.view(-1), seq_inputs.size(0)



def load_embedding_matrix(vocab, use_glove, glove_file_path="glove.840B.300d.txt"):
    if use_glove:
        embedding_dim = -1
    else:
        embedding_dim = 300

    embeddings = {}

    if not use_glove:
        print("not using glove, initializing embedding")
        embedding_matrix = torch.zeros(size=(len(vocab), embedding_dim))
        embedding_matrix = torch.nn.init.xavier_uniform_(embedding_matrix)
        # embedding_matrix = torch.normal()
    else:
        print("use embedding from path: ", glove_file_path)
        with open(glove_file_path, "r", encoding="utf8") as f:
            found_embedding_count = 0
            for token_embedding in f.readlines():
                token, *embedding = token_embedding.strip().split(" ")
                if token not in vocab:  # token is from embedding file
                    continue

                embedding = torch.tensor([float(e) for e in embedding], dtype=torch.float32)
                assert token not in embeddings
                assert embedding_dim < 0 or embedding_dim == len(embedding)
                found_embedding_count += 1
                embeddings[token] = embedding
                embedding_dim = len(embedding)
            not_found_embedding_count = len(vocab) - found_embedding_count
            print("Not found embedding count: ", not_found_embedding_count, " Ratio: ", not_found_embedding_count*1.0/len(vocab))
            print("Found embedding count: ", found_embedding_count, " Ratio: ", found_embedding_count * 1.0/len(vocab))
        all_embeddings = torch.stack(list(embeddings.values()), dim=0)

        embedding_mean = all_embeddings.mean()
        embedding_std = all_embeddings.std()
        # Randomly initialize embeddings
        embedding_matrix = torch.normal(embedding_mean, embedding_std, size=(len(vocab), embedding_dim))

        # Overwrite the embeddings we get from GloVe. The ones we don't find are left randomly initialized.
        for token, embedding in embeddings.items():
            embedding_matrix[vocab[token], :] = embedding

    # The padding token is explicitly initialized to 0.
    embedding_matrix[vocab["[pad]"]] = 0

    return embedding_matrix
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers import Tokenizer
import re


class SentenceDataset(Dataset):
    def __init__(self, sentence_data_path, token_level ="word", unk_cutoff = 3, tokenizer_path ="./tokenizer/", \
                 tokenizer_training_path ="./challenge-data/train_correct.txt", sequence_length = None):

        '''
        :param sentence_data_path:
        :param token_level:
        :param unk_cutoff:
        :param tokenizer_path:
        :param tokenizer_training_path:  Both used for train the tokenizer and the model
        :param sequence_length: length of sequence
        '''
        super().__init__()

        sentences = []
        # vocab_build_sents = []
        labels = []




        # intialize or load tokenizer
        if tokenizer_path == None:
            print("creating new tokenzier")
            assert tokenizer_training_path != None, "Must have valid text files for training"
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(files=tokenizer_training_path, min_frequency=unk_cutoff,
                            special_tokens=["[s]", "[pad]", "[/s]", "[unk]"])
            # tokenizer and sentences
            vocab = tokenizer.get_vocab()
            tokenizer._tokenizer.post_processor = BertProcessing(
                ("[/s]", tokenizer.token_to_id("[/s]")),
                ("[s]", tokenizer.token_to_id("[s]")),
            )
            tokenizer.save("./tokenizer/bpe.tokenizer.json")
        else:
            print("use old tokenizer")
            tokenizer = Tokenizer.from_file(tokenizer_path +  "bpe.tokenizer.json")
            vocab = tokenizer.get_vocab()
        self.tokenizer = tokenizer




        # initilize training data
        if re.match(".*.tsv", sentence_data_path) != None:
            print("loading corruption discriminator dataset")
            self.dataset_type = 0
            # sentences
            with open(sentence_data_path, "r") as f:

                reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                next(reader)  # Ignore header
                for row in reader:
                    # Each row contains a sentence and label (either 0 or 1)
                    sentence, label = row

                    sentences.append(sentence.strip())
                    # vocab_build_sents.append(sentence)
                    labels.append([int(label)])
        elif re.match(".*.txt", sentence_data_path) != None:
            print("loading corruption generator dataset")
            self.dataset_type = 1
            with open(sentence_data_path, "r") as f:
                for sentence in f.readlines():
                    if (len(sentence.strip())) == 0:
                        continue
                    sentences.append(sentence.strip())
        else:
            raise ValueError("Invalid file extension")



        # a sentence of tokens
        tokenized_sentences = [tokenizer.encode(sentence).tokens for sentence in sentences]

        indexed_sentences = [self.tokens_to_indices(tokenized_sentence) for tokenized_sentence in tokenized_sentences ]

        labels = torch.tensor(labels)


        self.vocab = vocab
        self.sentences = indexed_sentences
        self.labels = labels
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        if self.dataset_type == 0:
            return self.sentences[index], self.labels[index]
        elif self.dataset_type == 1:
            return self.sentences[index]
        else:
            print("get item error ")
            exit()
    def __len__(self):
        return len(self.sentences)


    def tokens_to_indices(self, tokens):
        """
        Converts tokens to indices.
        :param tokens: A list of tokens (strings)
        :return: A tensor of shape (n, 1) containing the token indices
        """
        indices = []
        for token in tokens:
            indices.append(self.tokenizer.token_to_id(token))
        return torch.tensor(indices)

    def indices_to_tokens(self, indices, token_level = "word"):
        """
        Converts indices to tokens and concatenates them as a string.
        :param indices: A tensor of indices of shape (n, 1), a list of (1, 1) tensors or a list of indices (ints)
        :return: The string containing tokens, concatenated by a space.
        """
        tokens = []
        for index in indices:
            if torch.is_tensor(index):
                index = index.item()
            token = self.tokenizer.id_to_token(index)
            tokens.append(token)
        return tokens
class GeneratorDataset(SentenceDataset):

    # TODO: would the length be the number of sequences rather than sentences??
    # def __len__(self):
    #     return len(self.)


    def __getitem__(self, index):
        assert self.sequence_length != None, "must define a sequence length for generator dataset"
        return self.sentences[index] + self.sequence_length, self.labels[index]
# class InferenceDataset(TrainingDataset):
#     def __init__(self, inference_file_path):
#         super.__init__()
#         assert inference_file_path != None, "Entering inference step, must have inference data file"
#
#         inf_f = open(inference_file_path, "r")
#         parallel_sents = []
#         for l in inf_f.readlines():
#             l = l.strip().split()
#             parallel_sents.append(l)
#
#         tokenizer = Tokenizer.from_file(tokenizer_path)
#
#         [tokenizer.encode(sentence).tokens for sentence in sentences];
#
#         # how to take   model(sentA) model(sentB)

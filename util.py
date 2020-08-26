import torch
from torch.utils.data import Dataset

import os
import pickle
import csv

from collections import Counter





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



class SentenceDataset(Dataset):
    def __init__(self, path, token_level = "word", unk_cutoff = 3, tokenizer_path = "./tokenizer/", \
                 training_file_path ="./challenge-data/train_20000.txt"):

        '''
        :param path:
        :param token_level:
        :param unk_cutoff:
        :param tokenizer_path:
        :param training_file_path:  Both used for train the tokenizer and the model

        '''
        super().__init__()

        sentences = []
        vocab_build_sents = []
        labels = []

        # sentences
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)  # Ignore header
            for row in reader:
                # Each row contains a sentence and label (either 0 or 1)
                sentence, label = row
                if token_level == "word":
                    # sentence = nltk.regexp_tokenize(sentence.strip(), pattern)
                    sentences.append(sentence.strip())
                    # if int(label) == 1:
                    vocab_build_sents.append(sentence)
                else:
                    sentences.append([ch for ch in sentence.strip()])
                    # if int(label) == 1:
                    #     correct_sents.append([ch for ch in sentence.strip()]) # add a stripped sentence (split latter)
                labels.append([int(label)])

        # intialize tokenizer
        if tokenizer_path == None:
            print("creating new tokenzier")
            assert training_file_path != None, "Must have valid text files for training"
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(files=training_file_path, min_frequency=unk_cutoff,
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
        # a sentence of tokens
        tokenized_sentences = [tokenizer.encode(sentence).tokens for sentence in sentences]

        indexed_sentences = [self.tokens_to_indices(tokenized_sentence) for tokenized_sentence in tokenized_sentences ]

        labels = torch.tensor(labels)


        self.vocab = vocab
        self.sentences = indexed_sentences
        self.labels = labels

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

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
#
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

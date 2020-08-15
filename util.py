import torch
from torch.utils.data import Dataset

import os
import pickle
import csv

from collections import Counter


def load_embedding_matrix(vocab, use_glove, glove_file_path="glove.6B.50d.txt"):
    if use_glove:
        embedding_dim = -1
    else:
        embedding_dim = 300

    embeddings = {}

    if not use_glove:
        print("not using glove")
        embedding_matrix = torch.zeros(size=(len(vocab), embedding_dim))
        embedding_matrix = torch.nn.init.xavier_uniform_(embedding_matrix)
        # embedding_matrix = torch.normal()
    else:
        print("use glove")
        with open(glove_file_path, "r", encoding="utf8") as f:
            for token_embedding in f.readlines():
                token, *embedding = token_embedding.strip().split(" ")

                if token not in vocab:
                    continue

                embedding = torch.tensor([float(e) for e in embedding], dtype=torch.float32)

                assert token not in embeddings
                assert embedding_dim < 0 or embedding_dim == len(embedding)

                embeddings[token] = embedding
                embedding_dim = len(embedding)

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


class SST2Dataset(Dataset):
    def __init__(self, path, vocab=None, reverse_vocab=None, token_level = "character", unk_cutoff = 3 ):
        super().__init__()

        sentences = []
        correct_sents = []
        labels = []
        import nltk
        pattern = r'''(?x) (?:[A-Z]\.)+  | \w+(?:-\w+)*  | \$?\d+(?:\.\d+)?%? | \.\.\. | [][.,;"'?():-_`] '''
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)  # Ignore header
            for row in reader:
                # Each row contains a sentence and label (either 0 or 1)
                sentence, label = row
                if token_level == "word":
                    sentence = nltk.regexp_tokenize(sentence.strip(), pattern)
                    sentences.append(sentence)
                    if int(label) == 1:
                        correct_sents.append(sentence)
                else:
                    sentences.append([ch for ch in sentence.strip()])
                    # if int(label) == 1:
                    #     correct_sents.append([ch for ch in sentence.strip()]) # add a stripped sentence (split latter)
                labels.append([int(label)])

        # Vocab maps tokens to indices
        if vocab is None:
            vocab = self._build_vocab(correct_sents,unk_cutoff , token_level=token_level)
            reverse_vocab = None

        # Reverse vocab maps indices to tokens
        if reverse_vocab is None:
            reverse_vocab = {index: token for token, index in vocab.items()}

        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        # a sentence of tokens
        if token_level == "word":
            indexed_sentences = [torch.tensor(self.tokens_to_indices(sentence)) for sentence in sentences]
        else:
            indexed_sentences = [torch.tensor(self.tokens_to_indices(sentence)) for sentence in sentences]

            # indexed_sentences = []
            # for sent in sentences:
            #     indexed_sentences.append(sent.split())


        labels = torch.tensor(labels)


        self.sentences = indexed_sentences
        self.labels = labels

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def _build_vocab(sentences, unk_cutoff=3, vocab_file_path="vocab.pkl", token_level = "word"):
        # Load cached vocab if existent
        if os.path.exists(vocab_file_path):
            with open(vocab_file_path, "rb") as f:
                return pickle.load(f)

        word_counts = Counter()

        print("building vocab......")

        count = 0
        threshold = 0.1

        # Count unique words (lower case)
        if token_level == "word":
            for sent in sentences:

                for token in sent: # here sent is a list of words
                    word_counts[token.lower()] += 1
                if count/len(sentences) > threshold:
                    print("building progress2: ", count/len(sentences) )
                    threshold += 0.1
                count += 1
        else:
            for sent in sentences:
                for token in sent: # here sent is a complete sentence
                    word_counts[token.lower()] += 1
                if count/len(sentences) > threshold:
                    print("building progress: ", count/len(sentences) )
                    threshold += 0.1
                count += 1
        # Special tokens: padding, beginning of sentence, end of sentence, and unknown word
        vocab = {"[pad]": 0, "[unk]": 1}
        token_id = 2

        print("Assigning id")
        # Assign a unique id to each word that occurs at least unk_cutoff number of times
        unk_count = 0
        import csv
        out_f = open("unk_output.tsv", "w")
        tsv_writer = csv.writer(out_f, delimiter='\t')
        tsv_writer.writerow(['unk', 'label'])
        for token, count in word_counts.items():
            if count >= unk_cutoff:
                vocab[token] = token_id
                token_id += 1
            else:
                tsv_writer.writerow([token, "0"])
                unk_count += 1
        print("Number of unk: ", unk_count)

        # Cache vocab
        with open(vocab_file_path, "wb") as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
        print("vocab size: ", len(vocab))
        return vocab

    def tokens_to_indices(self, tokens):
        """
        Converts tokens to indices.
        :param tokens: A list of tokens (strings)
        :return: A tensor of shape (n, 1) containing the token indices
        """
        indices = []

        unk_token = self.vocab["[unk]"]

        for token in tokens:
            indices.append(self.vocab.get(token.lower(), unk_token))

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
            token = self.reverse_vocab.get(index, "[unk]")
            if token == "[pad]":
                continue
            tokens.append(token)
        if token_level == "word":
            recovered_sent = " ".join(tokens)
        else:
            recovered_sent = "".join(tokens)
        return recovered_sent

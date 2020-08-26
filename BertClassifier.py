from util import *
from pathlib import Path
import math
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import torch
import numpy as np

from torch.utils.data import Dataset

class EsperantoDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer(
            "./esperberto-vocab.json",
            "./esperberto-merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        src_files = Path("./data/").glob("*-eval.txt") if evaluate else Path("./data/").glob("train_20000.txt")
        for src_file in src_files:
            print("🔥", src_file)
        lines = src_file.read_text(encoding="utf-8").splitlines()
        self.examples += [x.ids for x in tokenizer.encode_batch(lines)]


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])

def get_score(sentence):
    tokenize_input = tokenizer.encode(sentence).tokens
    print(tokenize_input)
    # tensor_input = torch.tensor(np.array([tokenizer.token_to_id(t) for t in tokenize_input]))
    # tensor_input = tensor_input[None, :]
    # print(tensor_input.shape)
    # print(tensor_input)

    # tokenize_input = tokenizer2.tokenize(sentence)
    # tensor_input = torch.tensor([tokenizer2.convert_tokens_to_ids(tokenize_input)])
    # print(tensor_input.shape)
    # print(tensor_input)
    # predictions = bertMaskedLM(tensor_input)
    # loss_fct = torch.nn.CrossEntropyLoss()
    # loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
    # return math.exp(loss)

if __name__ == '__main__':
    # train_data = SST2Dataset("./challenge-data/train_20000.txt", token_level="word", unk_cutoff=3)

    # bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files = "./challenge-data/train_20000.txt", min_frequency=2, special_tokens=["<s>","<pad>","</s>","<unk>"])
    # tokenizer.save_model(".", "esperberto")


    # tokenizer = ByteLevelBPETokenizer(
    #     "./esperberto-vocab.json",
    #     "./esperberto-merges.txt",
    # )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    # tokenizer.enable_truncation(max_length=512)




    tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased')
    data_f = open("./challenge-data/train_20000_corrupted.txt", "r")



    lines = data_f.readlines()
    count = 0
    for l in lines:
        # l = "All author have seen the manuscript and approved to submit to your journal ."
        print("t1 ------------------------")
        print(tokenizer.encode(l).tokens)
        token_l = tokenizer.encode(l).tokens
        tokenizer.
        for t in token_l:
            print(tokenizer.token_to_id(t))
        print(get_score(l))
        print(count)
        if count > 2:
            break
        count += 1
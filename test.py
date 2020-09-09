from hw4_a6 import *
from util import *
import torch
from torch.utils.data import DataLoader


from train import generate_sampler
from tokenizers import Tokenizer
import argparse
import torch.nn as nn
from util import collate_fn
from model import RNNBinaryClassificationModel, TRAINING_BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,\
                VAL_BATCH_SIZE, TEST_BATCH_SIZE
from tqdm import tqdm



def test(device, old_model_state, old_embedding, dataset):


    n = len(dataset)
    # test_sampler = generate_sampler(n, shuffle_dataset = False)


    test_loader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, collate_fn = collate_fn)

    print("Data examples:")
    num_examples = 10
    assert TEST_BATCH_SIZE % 2 == 0, "evaluation must "
    # random_indices = torch.randperm(len(train_dataset))[:num_examples].tolist()
    indices = torch.arange(0, num_examples)
    for index in indices:
        sequence_indices, label = dataset.sentences[index], dataset.labels[index]
        correctness = "Positive" if label == 1 else "Negative"
        sequence = dataset.indices_to_tokens(sequence_indices, token_level)
        print(f"Correct: {correctness}. Sentence: {sequence}")

    # print(old_embedding)
    # exit()
    model = RNNBinaryClassificationModel(old_embedding, device).cuda()
    model.load_state_dict(old_model_state)
    model = nn.DataParallel(model, device_ids = [0,1])

    model.to(device)

    tqdm_test_loader = tqdm(test_loader)
    test_correct = 0
    test_seqs = 0
    model.eval()
    for batch_idx, batch in enumerate(tqdm_test_loader):
        sentences_batch, labels_batch = batch
        sentences_batch = sentences_batch.to(device)
        labels_batch = labels_batch.to(device)
        print("labels batch: ", labels_batch)
        with torch.no_grad():
            # Make predictions
            logits = model(sentences_batch)
            print("logits: ", logits)
            pairLogits = toPairLogits(logits)

            correct = model.module.accuracy(pairLogits, labels_batch).item() * len(pairLogits)
            test_correct += correct
            test_seqs += len(sentences_batch)
            print(f"batch accuracy: {correct/TEST_BATCH_SIZE:.4f} ")
            tqdm_test_loader.set_description_str(
                f"[Acc]: {test_correct / test_seqs: .4f}"
            )

    test_accuracy = test_correct/test_seqs
    print(f"[Test accuracy]: {test_accuracy:.4f}")


def toPairLogits(t):
    t1 = t[::2]
    t2 = t[1::2]

    comp_t1 = (t1 > t2).int()
    comp_t2 = (t2 > t1).int()

    new_t = torch.zeros_like(t)

    new_t[comp_t1.nonzero() * 2] = 1
    new_t[comp_t2.nonzero() * 2  +1] = 1
    return new_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    tokenizer_path = "./tokenizer/bpe.tokenizer.json"
    checkpoint = torch.load("./model/best_model.pt")
    old_embedding =  checkpoint['embedding_matrix']
    old_model_state = checkpoint['model_stat_dict']
    token_level = "word"
    unk_cutoff = 3
    dataset = SentenceDataset("./challenge-data/train_20000.tsv", unk_cutoff=unk_cutoff, tokenizer_path = tokenizer_path) # can be either test or fine-tune traiing

    device = 0
    test(device, old_model_state, old_embedding, dataset)

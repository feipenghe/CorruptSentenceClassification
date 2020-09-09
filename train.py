import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import numpy as np
import torch.nn as nn
from util import SentenceDataset, load_embedding_matrix, collate_fn
from model import RNNBinaryClassificationModel, TRAINING_BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,\
                VAL_BATCH_SIZE



def train(device, use_glove, token_level="word", unk_cutoff = 3 ):

    # Load datasets
    train_dataset = SentenceDataset("./challenge-data/train.tsv", tokenizer_path = None, token_level = token_level, unk_cutoff = unk_cutoff)
    n =len(train_dataset)



    train_sampler, val_sampler = generate_sampler(n)

    print("loading data...")
    # Create data loaders for creating and iterating over batches
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, collate_fn=collate_fn, sampler=train_sampler)
    # train_loader = DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=VAL_BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)

    # Print out some random examples from the data
    print("Data examples:")
    num_examples = 20
    random_indices = torch.randperm(len(train_dataset))[:num_examples].tolist()
    for index in random_indices:
        sequence_indices, label = train_dataset.sentences[index], train_dataset.labels[index]
        correctness = "Positive" if label == 1 else "Negative"
        sequence = train_dataset.indices_to_tokens(sequence_indices, token_level)
        print(f"Correct: {correctness}. Sentence: {sequence}")

    embedding_matrix = load_embedding_matrix(train_dataset.vocab, use_glove)

    model = RNNBinaryClassificationModel(embedding_matrix, device).cuda()
    print("Data parallel!!!!!!!!")
    model = nn.DataParallel(model, device_ids = [0,1])

    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05*LEARNING_RATE)
    import sys
    best_val_loss = sys.maxsize
    best_val_acc = None # come with best validation loss
    best_train_loss = None
    best_train_acc = None

    val_acc_l = []
    patience = 5
    for epoch in range(NUM_EPOCHS):
        # Total loss across train data
        train_loss = 0.
        # Total number of correctly predicted training labels
        train_correct = 0
        # Total number of training sequences processed
        train_seqs = 0

        tqdm_train_loader = tqdm(train_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        model.train()
        for batch_idx, batch in enumerate(tqdm_train_loader):
            sentences_batch, labels_batch = batch
            sentences_batch = sentences_batch.to(device)
            labels_batch = labels_batch.to(device)
            # Make predictions
            logits = model(sentences_batch)
            # Compute loss and number of correct predictions
            loss = model.module.loss(logits, labels_batch)

            correct = model.module.accuracy(logits, labels_batch).item() * len(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics and update status
            train_loss += loss.item()
            train_correct += correct
            train_seqs += len(sentences_batch)
            print(f"\t\t\t\t\t\t\t\t\t\t\t\t[Batch idx: {batch_idx:d}]  Batch accuracy: {correct/TRAINING_BATCH_SIZE :.4f}")
            tqdm_train_loader.set_description_str(
                f"[Epoch {epoch:d}] [Loss]: {train_loss / (batch_idx + 1):.4f} [Acc]: {train_correct / train_seqs:.4f}")
        print()

        avg_train_loss = train_loss / len(tqdm_train_loader)
        train_accuracy = train_correct / train_seqs
        print(f"[Training Loss]: {avg_train_loss:.4f} [Training Accuracy]: {train_accuracy:.4f}")

        print("Validating")
        # Total loss across validation data
        val_loss = 0.
        # Total number of correctly predicted validation labels
        val_correct = 0
        # Total number of validation sequences processed
        val_seqs = 0

        tqdm_val_loader = tqdm(val_loader)

        model.eval()
        for batch_idx, batch in enumerate(tqdm_val_loader):
            sentences_batch, labels_batch = batch
            sentences_batch = sentences_batch.to(device)
            labels_batch = labels_batch.to(device)
            with torch.no_grad():
                # Make predictions
                logits = model(sentences_batch)

                # Compute loss and number of correct predictions and accumulate metrics and update status
                val_loss += model.module.loss(logits, labels_batch).item()
                val_correct += model.module.accuracy(logits, labels_batch).item() * len(logits)
                val_seqs += len(sentences_batch)
                tqdm_val_loader.set_description_str(
                    f"[Loss]: {val_loss / (batch_idx + 1):.4f} [Acc]: {val_correct / val_seqs:.4f}")
        print()

        avg_val_loss = val_loss / len(tqdm_val_loader)
        val_accuracy = val_correct / val_seqs
        print(f"[Validation Loss]: {avg_val_loss:.4f} [Validation Accuracy]: {val_accuracy:.4f}")
        val_acc_l.append(val_accuracy)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy  # come with best validation loss
            best_train_loss = avg_train_loss
            best_train_acc = train_accuracy
            best_epoch = epoch
            torch.save({'epoch': best_epoch,
                        'model_stat_dict' :model.module.state_dict(),
                        'embedding_matrix' : model.module.embedding.weight.data}, "./model/best_model.pt")
        if len(val_acc_l) > patience + 1:
            if val_acc_l[-1] < val_acc_l[-patience]:
                break
    print("best epoch : ", epoch)
    print("best_val_loss: ", best_val_loss)
    print("best_val_acc: ", best_val_acc)
    print("best_train_loss: ", best_train_loss)
    print("best_train_acc: ", best_train_acc)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default= 0)
    ap.add_argument("--use_glove", default = True, type = eval, choices = [True, False])
    ap.add_argument("--token_level", default = "character")
    ap.add_argument("--unk_cutoff", default= 3)
    args = ap.parse_args()

    train(int(args.gpu), bool(args.use_glove), args.token_level, unk_cutoff= int(args.unk_cutoff))

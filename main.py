import argparse
import copy
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model import DeepAveragingNetwork
from torch.utils.data.sampler import SubsetRandomSampler

from hw4_a6 import collate_fn
from train import generate_sampler
from tqdm import tqdm
from util import SST2Dataset
def set_seed(seed):
    """ Set various random seeds for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, train_loader, criterion, optimizer, padding_index):
    """ Train a model for one epoch.

    Arguments:
        model: model to train
        iterator: iterator over one epoch of data to train on
        criterion: loss, to be used for training
        optimizer: the optimizer to use for training
        padding_index: which token ID is padding, for model

    Returns:
        average of criterion across the iterator
    """

    # put model back into training mode
    model.train()

    epoch_loss = 0.0
    tqdm_train_loader = tqdm(train_loader)
    for batch in tqdm_train_loader:
        sentences, labels = batch
        optimizer.zero_grad()

        logits = model(sentences, padding_index)
        labels = labels.squeeze()
        loss = criterion(logits, labels)

        # TODO: implement L2 loss here
        # Note: model.parameters() returns an iterator over the parameters
        # (which are tensors) of the model

        # L2 by the batch
        if args.L2:
            L2_squared = torch.zeros(1)
            for parameter in model.parameters():
                # possibly sum of L2 for each parameter tensor
                parameter_squared = torch.mul(parameter, parameter)
                L2_squared += torch.sum(parameter_squared)
            # L2 = math.sqrt(L2_squared)
            L2 = L2_squared
            # print(L2)
        else:
            L2 = 0

        regularized_loss = loss + 1e-4*L2

        # backprop regularized loss and update parameters
        regularized_loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def evaluate(model, val_loader, criterion, padding_index):
    """ Evaluate a model.

    Arguments:
        model: model to evaluate
        iterator: iterator over data to evaluate on
        criterion: metric for evaluation
        padding_index: which token ID is padding, for model

    Returns:
        average of criterion across the iterator
    """
    # put model in eval mode
    model.eval()
    epoch_loss = 0.0
    for batch in val_loader:
        sentences, labels = batch
        logits = model(sentences, padding_index)

        labels = labels.squeeze()
        loss = criterion(logits, labels)
        epoch_loss += loss.item()
    return epoch_loss / len(val_loader)


def accuracy(logits, labels):
    """Computes accuracy of model outputs given true labels.

    Arguments:
        logits: (batch_size, num_classes) tensor of logits from a model
        labels: (batch_size) tensor of class labels

    Returns:
        percentage of correct predictions, where the prediction is taken to be
        the class with the highest logit / probability
    """
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).float()
    return correct.mean()


def main(args):
    """Main method: gathers and splits train/dev/test data, builds and then
    trains and evaluates a model.
    """

    # get start time
    start = time.time()

    # set all random seeds
    set_seed(args.seed)

    # get iterators for data
    train_data = SST2Dataset("./challenge-data/train.tsv", token_level="word", unk_cutoff=3)


    # sampling
    n = len(train_data)
    train_sampler, val_sampler = generate_sampler(n)

    print("loading data...")
    TRAINING_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 128
    # Create data loaders for creating and iterating over batches
    train_loader = DataLoader(train_data, batch_size=TRAINING_BATCH_SIZE, collate_fn=collate_fn,
                              sampler=train_sampler)
    # train_loader = DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(train_data, batch_size=VAL_BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)




    # build model
    model = DeepAveragingNetwork(
        len(train_data.vocab), args.embedding_dim, args.hidden_dim, 2)

    # set up optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Epoch \t Train loss \t Dev loss \n{'-'*40}")
    best_loss = math.inf
    best_model = None
    best_epoch = None
    padding_index = args.padding_index
    # main training loop
    loss_l = []
    loss_ix = -1
    for epoch in range(args.num_epochs):
        # train for one epoch
        epoch_train_loss = train_one_epoch(
            model, train_loader, criterion, optim, padding_index)
        # evaluate on dev set
        dev_loss = evaluate(model, val_loader, criterion, padding_index)
        print(f"{epoch} \t {epoch_train_loss:.5f} \t {dev_loss:.5f}")
        loss_l.append(dev_loss)
        loss_ix  += 1
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        if args.patience and epoch >= args.patience-1:  # it defines when to start consdier patience
            # TODO: implement early stopping here.
            # Note: you may need to touch some code outside of this if
            # statement.
            if loss_l[loss_ix] > loss_l[loss_ix-args.patience]:
                break

    print(f"Evaluating best model (from epoch {best_epoch}) on test set.")
    test_loss = evaluate(best_model, val_loader, criterion, padding_index)
    test_accuracy = evaluate(best_model, val_loader, accuracy, padding_index)
    print(f"test loss: {test_loss}\ntest accuracy: {test_accuracy}")

    end = time.time()
    print(f"total time: {end - start}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    # training arguments
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=572)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--L2', action="store_true")
    # data arguments
    parser.add_argument('--data_dir', type=str, default='/dropbox/19-20/572/hw9/data')
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--padding_index', type=int, default=0)
    args = parser.parse_args()

    main(args)

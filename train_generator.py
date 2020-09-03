from model import CorruptionGenerator, LEARNING_RATE
from torch.utils.data import DataLoader
from util import seq_collate_fn, SentenceDataset
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
def train_generator(device, model, unk_cutoff = 3, bptt= 3):
    # TODO: load txt training dataset
    SENT_BATCH_SIZE = 16

    model.train()

    n = len(dataset)
    # test_sampler = generate_sampler(n, shuffle_dataset = False)

    train_loader = DataLoader(dataset, batch_size=SENT_BATCH_SIZE, collate_fn=seq_collate_fn)

    optimizer = optim.RMSprop(model.parameters(), lr =LEARNING_RATE)
    tqdm_train_loader = tqdm(train_loader)
    train_loss = 0.
    train_correct = 0
    train_seqs = 0


    SEQ_BATCH_SIZE = len(train_loader)


    first_batch = True
    for seq_batch_idx, seq_batch in enumerate(tqdm_train_loader):
        # print("batch idx:   ", seq_batch_idx)
        if first_batch:
            seq_batch, labels_batch, batch_size = seq_batch
            # h0, c0 = model.module.init_state(batch_size)
            # print("batch size: ")
            h0, c0 = model.init_state(batch_size)
            state = (h0,c0)
        else:
            seq_batch, labels_batch, _ = seq_batch
        seq_batch = seq_batch.to(device)
        labels_batch = labels_batch.to(device)
        # print("seq batch shape: ", seq_batch.shape)

        # try:
        logits, state = model(seq_batch, state)
        # except RuntimeError as re:
        #     print(state.size())
        #     print(re)
        #     exit(0)
        # loss = model.module.loss(logits, labels_batch)
        loss = model.loss(logits, labels_batch)

        # correct = model.module.accuracy(logits, labels_batch).item() * len(logits)
        correct = model.accuracy(logits, labels_batch).item() * len(logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += correct
        train_seqs += len(seq_batch)
        print(
            f"\t\t\t\t\t\t\t\t\t\t\t\t[Batch idx: {seq_batch_idx:d}]  Batch accuracy: {correct / batch_size :.4f}")











if __name__ == '__main__':
    tokenizer_path = "./tokenizer/"

    checkpoint = torch.load("./model/best_model.pt")
    old_embedding = checkpoint['embedding_matrix']

    dataset = SentenceDataset("./challenge-data/train_200000.txt",
                              unk_cutoff=3, tokenizer_path = tokenizer_path)  # can be either test or fine-tune traiing

    model = CorruptionGenerator(old_embedding).cuda()
    # model = nn.DataParallel(model, device_ids = [0, 1])
    train_generator(0, model)
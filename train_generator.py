from model import CorruptionDiscriminator, EncoderRNN, DecoderRNN
from torch.utils.data import DataLoader
from util import seq_collate_fn, SentenceDataset, generate_sampler, collate_fn
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn

def repackage_hidden(h):
    """
    Wraps hidden states in new tensors to detach them from their history
    :param h: hidden state
    :return:
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h) # for cases of (h, c)


def train_generator(device, generator_encoder, generator_decoder, discriminator, dataset):
    # TODO: load txt training dataset
    SENT_BATCH_SIZE = 400



    n = len(dataset)
    # initialization
    train_sampler, val_sampler = generate_sampler(n)
    train_loader = DataLoader(dataset, batch_size=SENT_BATCH_SIZE, collate_fn=collate_fn, sampler= train_sampler)
    val_loader = DataLoader(dataset, batch_size=SENT_BATCH_SIZE, collate_fn=collate_fn, sampler= val_sampler)
    LEARNING_RATE = 5e-4
    encoder_optimizer = optim.RMSprop(generator_encoder.parameters(), lr =LEARNING_RATE,  weight_decay=0.05*LEARNING_RATE)
    decoder_optimizer = optim.RMSprop(generator_decoder.parameters(), lr =LEARNING_RATE,  weight_decay=0.05*LEARNING_RATE)
    # fake_discriminator_optimizer = optim.RMSprop(discriminator_model.parameters(), lr =LEARNING_RATE)
    NUM_EPOCH = 20

    val_acc_l = []
    patience = 5
    for epoch in range(NUM_EPOCH):
        train_loss = 0.
        train_correct = 0
        train_seqs = 0
        tqdm_train_loader = tqdm(train_loader)
        generator_encoder.train()
        generator_decoder.train()
        # discriminator_model.train()
        for sent_batch_idx, batch in enumerate(tqdm_train_loader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            sent_batch, labels_batch = batch

            # trim one data when there are odd number of data
            # for stable hidden size
            if len(sent_batch) % 2 != 0: # it happens at the end of sent batch
                sent_batch = sent_batch[:-1]
                labels_batch = labels_batch[:-1]
            print("sent batch:  ", sent_batch.shape)

            loss = 0

            # 1. encode
            encoder_hidden = generator_encoder.module.init_state(len(sent_batch)) # initial state every batch

            sent_batch = sent_batch.to(device)
            labels_batch = labels_batch.to(device)

            encoder_output, encodder_hidden = generator_encoder(sent_batch, encoder_hidden)
            decoder_input = dataset.tokens_to_indices(['[s]']).repeat(len(sent_batch)).unsqueeze(dim=1).cuda() # TODO: a batch of start symbol
            decoder_hidden = encodder_hidden # take the last hidden layer

            import random
            seq_len = 6
            start = random.randint(1,sent_batch.size(1)-seq_len)
            end = min(start + seq_len, sent_batch.size(1))
            # step = random.randint(2, 7)
            decoder_input = sent_batch[:, start].unsqueeze(dim=1)
            # 2. decode
            for di in range(start, end):
                decoder_output, decoder_hidden = generator_decoder(decoder_input, decoder_hidden)
                # loss += generator_criterion(decoder_output, sent_batch[:, di]) # encourage to be the same, TODO: it has risk being the same
                modified_batch = sent_batch.clone()
                if decoder_output.get_device() == 0:
                    modified_batch[:len(sent_batch)//2, di] = torch.argmax(decoder_output, dim=2).squeeze()
                else:
                    modified_batch[len(sent_batch) // 2:, di] = torch.argmax(decoder_output, dim=2).squeeze()

                # discriminator doesn't do anything with it
                # discriminator loss is not bp to encoder decoder
                # sent batch is not correct way to measure the amount of memory ->
                # with torch.no_grad():
                discriminator_logits = discriminator_model(modified_batch)
                correct = discriminator_model.module.accuracy(discriminator_logits, labels_batch).item() * len(discriminator_logits)
                train_correct += correct
                # two losses
                discriminator_loss = discriminator.module.inv_loss(discriminator_logits, labels_batch) # if different, few er loss. if same, no loss.
                print("discriminator loss: ", discriminator_loss,  "    accuracy: ", correct*1.0/len(labels_batch))
                loss += discriminator_loss
                # discriminator_loss.detach() # want it not to do with discriminator model
                generator_loss = generator_decoder.module.inv_loss( decoder_output.squeeze(), sent_batch[:,di].view(-1))
                print("generator loss:  ", generator_loss)

                loss += generator_loss

                # prepare for the next decoder input
                decoder_input = sent_batch[:, di].unsqueeze(dim=1)

            loss.backward()


            encoder_optimizer.step()
            decoder_optimizer.step()
            # fake_discriminator_optimizer.step()


            train_loss += loss.item()

            # train_seqs +=  sent_batch.size(0) * sent_batch.size(1)
            train_seqs += sent_batch.size(0) * (end-start)
            tqdm_train_loader.set_description_str(
                f"[Epoch {epoch:d}] [Loss]: {train_loss / (sent_batch_idx + 1):.4f} [Discriminator Acc]: {train_correct / train_seqs:.4f}")
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
        import sys
        best_val_loss = sys.maxsize
        generator_decoder.eval()
        for batch_idx, batch in enumerate(tqdm_val_loader):
            sent_batch, labels_batch = batch
            sent_batch = sent_batch.to(device)
            labels_batch = labels_batch.to(device)
            if len(sent_batch) % 2 != 0: # it happens at the end of sent batch
                sent_batch = sent_batch[:-1]
                labels_batch = labels_batch[:-1]
            state = generator_decoder.module.init_state(len(sent_batch))
            with torch.no_grad():
                # Make predictions

                decoder_input = dataset.tokens_to_indices(['[s]']).repeat(len(sent_batch)).unsqueeze(
                    dim=1).cuda()  # TODO: a batch of start symbol
                # Compute loss and number of correct predictions and accumulate metrics and update status
                for di in range(sent_batch.size(1)):
                    decoder_output, _ = generator_decoder(decoder_input,
                                                  state)  # pass the last state during training and doesn't make any change
                    print("logits shape: ", decoder_output.shape)
                    print("target shape: ", sent_batch[:,di].shape)
                    val_loss += generator_decoder.module.inv_loss( decoder_output.squeeze(), sent_batch[:,di].view(-1))

                    val_correct += generator_decoder.module.accuracy(torch.argmax(decoder_output.squeeze(), dim=1 ), sent_batch[:,di].view(-1)).item() * len(decoder_output)
                    decoder_input = sent_batch[:, di].unsqueeze(dim=1)
                val_seqs += sent_batch.size(0) * sent_batch.size(1)  # len(sent_batch)

                tqdm_val_loader.set_description_str(
                    f"[Validation Loss]: {val_loss / (batch_idx + 1):.4f} [Decoder Acc]: {val_correct / val_seqs:.4f}")
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
                        'encoder_embedding': generator_encoder.module.embedding.weight.data,
                        'decoder_embedding': generator_decoder.module.embedding.weight.data,
                        'encoder_model_stat_dict': generator_encoder.module.state_dict(),
                        'decoder_model_stat_dict': generator_decoder.module.state_dict()},
                        "./model/best_generator_model.pt")
        # if len(val_acc_l) > patience + 1:
        #     if val_acc_l[-1] > val_acc_l[-patience]:
        #         break
    print("best epoch : ", best_epoch)
    print("best_val_loss: ", best_val_loss)
    print("best_val_acc: ", best_val_acc)
    print("best_train_loss: ", best_train_loss)
    print("best_train_acc: ", best_train_acc)
    print("worst val acc: ",  val_acc_l[-patience])







if __name__ == '__main__':
    tokenizer_path = "./tokenizer/"

    discriminator_checkpoint = torch.load("./model/best_model.pt")
    old_embedding = discriminator_checkpoint['embedding_matrix']
    old_model_state = discriminator_checkpoint['model_stat_dict']
    discriminator_model = CorruptionDiscriminator(old_embedding, 0).cuda()
    discriminator_model.load_state_dict(old_model_state)

    # generator_checkpoint = torch.load("/model/best_generator_model.pt")
    dataset = SentenceDataset("./challenge-data/train_5000.tsv",
                              unk_cutoff=3, tokenizer_path = tokenizer_path)  # can be either test or fine-tune traiing

    hidden_size = 100
    output_size = old_embedding.size(0)
    generator_encoder = EncoderRNN(old_embedding, hidden_size).cuda()
    generator_decoder = DecoderRNN(hidden_size, output_size).cuda()


    # data parallel
    discriminator_model = nn.DataParallel(discriminator_model, device_ids=[0, 1])
    generator_encoder = nn.DataParallel(generator_encoder, device_ids=[0, 1])
    generator_decoder = nn.DataParallel(generator_decoder, device_ids=[0, 1])

    train_generator(0, generator_encoder, generator_decoder, discriminator_model, dataset)
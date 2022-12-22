import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from bert_dataset import BertDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import load_from_disk
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import wandb

def format_time(elapsed):
    # Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

def flat_f1(logits, label_ids):
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    return f1_score(labels_flat, pred_flat, average='macro')


def train(model, tokenizer, train_dataloader, validation_dataloader, epochs, optimizer, scheduler, device, model_save_path, seed):
    # set the seed value all over the place to make this reproducible.
    seed_val = seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # move model to device
    model.to(device)

    # trainig loop
    for epoch_i in range(0, epochs):
        model.train()
        # Measure how long the training epoch takes.
        t0 = time.time()
        # reset the total loss for this epoch.
        total_train_loss = 0
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            output = model(tokenizer(batch))
            loss = output.loss
            logits = output.logits
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            total_train_loss += loss.item()
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print(f"  Average training loss: {avg_train_loss}")
        print(f"  Training epoch took: {training_time}")
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        model.eval()
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_f1 = 0
        total_eval_loss = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            output = model(tokenizer(batch))
            loss = output.loss
            logits = output.logits
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = batch['labels'].cpu().numpy()
            # add the loss to the total loss for this epoch.
            total_eval_loss += loss.item()
            # Calculate the accuracy for this batch of test sentences.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            total_eval_f1 += flat_f1(logits, label_ids)
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_f1 = total_eval_f1 / len(validation_dataloader)
        print("  F1: {0:.2f}".format(avg_val_f1))
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))

        wandb.log({"epoch": epoch_i, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": avg_val_accuracy, "val_f1": avg_val_f1})
        print("  Validation took: {:}".format(validation_time))
        # Save the model
        torch.save(model.state_dict(), model_save_path)
        print("Saving model to %s" % model_save_path)
        print("")
    print("Training complete!")
    return model


if __name__ == "__main__":
    # load the dataset
    books_train_df = load_from_disk('data/books_balanced_train_dataset')
    books_validation_df = load_from_disk('data/books_balanced_validation_dataset')
    dvd_train_df = load_from_disk('data/movies_balanced_train_dataset')
    dvd_validation_df = load_from_disk('data/movies_balanced_validation_dataset')
    electronics_train_df = load_from_disk('data/electronics_balanced_train_dataset')
    electronics_validation_df = load_from_disk('data/electronics_balanced_validation_dataset')
    kitchen_train_df = load_from_disk('data/kitchen_balanced_train_dataset')
    kitchen_validation_df = load_from_disk('data/kitchen_balanced_validation_dataset')

    # load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    # define the datasets
    books_train_df = BertDataset(books_train_df, tokenizer, 128)
    books_validation_df = BertDataset(books_validation_df, tokenizer, 128)
    dvd_train_df = BertDataset(dvd_train_df, tokenizer, 128)
    dvd_validation_df = BertDataset(dvd_validation_df, tokenizer, 128)
    electronics_train_df = BertDataset(electronics_train_df, tokenizer, 128)
    electronics_validation_df = BertDataset(electronics_validation_df, tokenizer, 128)
    kitchen_train_df = BertDataset(kitchen_train_df, tokenizer, 128)
    kitchen_validation_df = BertDataset(kitchen_validation_df, tokenizer, 128)

    # create the dataloader
    books_train_dataloader = DataLoader(books_train_df, sampler=RandomSampler(books_train_df), batch_size=32)
    books_validation_dataloader = DataLoader(books_validation_df, sampler=SequentialSampler(books_validation_df), batch_size=32)
    dvd_train_dataloader = DataLoader(dvd_train_df, sampler=RandomSampler(dvd_train_df), batch_size=32)
    dvd_validation_dataloader = DataLoader(dvd_validation_df, sampler=SequentialSampler(dvd_validation_df), batch_size=32)
    electronics_train_dataloader = DataLoader(electronics_train_df, sampler=RandomSampler(electronics_train_df), batch_size=32)
    electronics_validation_dataloader = DataLoader(electronics_validation_df, sampler=SequentialSampler(electronics_validation_df), batch_size=32)
    kitchen_train_dataloader = DataLoader(kitchen_train_df, sampler=RandomSampler(kitchen_train_df), batch_size=32)
    kitchen_validation_dataloader = DataLoader(kitchen_validation_df, sampler=SequentialSampler(kitchen_validation_df), batch_size=32)

    # train the model
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 10
    num_trainings = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in ['books', 'dvd', 'electronics', 'kitchen']:
        print('Training on %s dataset' % dataset)
        train_dataloader = eval('%s_train_dataloader' % dataset)
        validation_dataloader = eval('%s_validation_dataloader' % dataset)
        for i in range(num_trainings):
            model_save_path = 'model_%s.bin' % dataset
            seed = i
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)
            train(model, tokenizer, train_dataloader, validation_dataloader, epochs, optimizer, scheduler, device, model_save_path, seed)



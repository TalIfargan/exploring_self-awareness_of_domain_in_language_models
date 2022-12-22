import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

# build train and validation dataloaders from Amazon Reviews (2018) dataset for binary classification
# devide the dataset to different product categories and save the dataset for the following categories:
# books, DVDs, electronics and kitchen appliances
from datasets import load_dataset
train_dataset = load_dataset('amazon_reviews_multi', 'en', split='train')
books_train_dataset = train_dataset.filter(lambda example: example['product_category'] == 'Books')
dvd_train_dataset = train_dataset.filter(lambda example: example['product_category'] == 'DVD')
electronics_train_dataset = train_dataset.filter(lambda example: example['product_category'] == 'Electronics')
kitchen_train_dataset = train_dataset.filter(lambda example: example['product_category'] == 'Kitchen')

# set the label to 0 for negative reviews and 1 for positive reviews if the rating is greater than 3 stars, drop neutral reviews
books_train_dataset = books_train_dataset.map(lambda example: {'label': 1 if example['star_rating'] > 3 else 0})
dvd_train_dataset = dvd_train_dataset.map(lambda example: {'label': 1 if example['star_rating'] > 3 else 0})
electronics_train_dataset = electronics_train_dataset.map(lambda example: {'label': 1 if example['star_rating'] > 3 else 0})
kitchen_train_dataset = kitchen_train_dataset.map(lambda example: {'label': 1 if example['star_rating'] > 3 else 0})

# balance the dataset to have the same number of positive and negative reviews for each category make the dataset as big as possible
books_balanced_train_dataset = books_train_dataset.filter(lambda example: example['label'] == 1).shuffle(seed=42)
books_balanced_train_dataset = books_balanced_train_dataset.concatenate(books_train_dataset.filter(lambda example: example['label'] == 0).shuffle(seed=42).select(range(books_balanced_train_dataset.dataset_size)))
dvd_balanced_train_dataset = dvd_train_dataset.filter(lambda example: example['label'] == 1).shuffle(seed=42)
dvd_balanced_train_dataset = dvd_balanced_train_dataset.concatenate(dvd_train_dataset.filter(lambda example: example['label'] == 0).shuffle(seed=42).select(range(dvd_balanced_train_dataset.dataset_size)))
electronics_balanced_train_dataset = electronics_train_dataset.filter(lambda example: example['label'] == 1).shuffle(seed=42)
electronics_balanced_train_dataset = electronics_balanced_train_dataset.concatenate(electronics_train_dataset.filter(lambda example: example['label'] == 0).shuffle(seed=42).select(range(electronics_balanced_train_dataset.dataset_size)))
kitchen_balanced_train_dataset = kitchen_train_dataset.filter(lambda example: example['label'] == 1).shuffle(seed=42)
kitchen_balanced_train_dataset = kitchen_balanced_train_dataset.concatenate(kitchen_train_dataset.filter(lambda example: example['label'] == 0).shuffle(seed=42).select(range(kitchen_balanced_train_dataset.dataset_size)))

# save the balanced datasets
books_balanced_train_dataset.save_to_disk('data/books_balanced_train_dataset')
dvd_balanced_train_dataset.save_to_disk('data/dvd_balanced_train_dataset')
electronics_balanced_train_dataset.save_to_disk('data/electronics_balanced_train_dataset')
kitchen_balanced_train_dataset.save_to_disk('data/kitchen_balanced_train_dataset')

# train the model
from transformers import AdamW, get_linear_schedule_with_warmup
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = 'model.bin'
seed = 42
train(model, tokenizer, train_dataloader, validation_dataloader, epochs, optimizer, scheduler, device, model_save_path, seed)


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
        nb_eval_steps = 0
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




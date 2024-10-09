import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.optim import AdamW
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import precision_score, recall_score, f1_score
best_f1_path = "model_best/VPFinder_head16_multi_f1.txt"
best_model_path = "model_best/VPFinder_head16_multi_model.pth"

# 对按照CWE顶层关系分配样本后的数据集

# Load data
df = pd.read_csv("data/dataset_240904.csv", low_memory=False)

# Split data into train and test sets
df_train, df_test = np.split(df, [int(.9 * len(df))])

# Load pre-trained tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("./mycodebert")
tokenizer2 = BertTokenizer.from_pretrained("./mybert")
labels = {'positive': 1, 'negative': 0}

class MultiClassDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = [tokenizer2(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in dataframe['text']]
        self.messages = [tokenizer2(message, padding='max_length', max_length=256, truncation=True, return_tensors="pt") for message in dataframe['message']]
        self.patch_del = [tokenizer1.encode_plus(str(patch), padding='max_length', max_length=256, truncation=True, return_tensors="pt") for patch in dataframe['del_patch']]
        self.patch_add = [tokenizer1.encode_plus(str(patch), padding='max_length', max_length=256, truncation=True, return_tensors="pt") for patch in dataframe['add_patch']]
        self.labels = [int(cwe_id) - 1 for cwe_id in dataframe['CWE_ID']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def get_batch_labels(self, idx):
        print(self.labels[idx])
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_del(self, idx):
        return self.patch_del[idx]

    def get_batch_add(self, idx):
        return self.patch_add[idx]

    def get_batch_messages(self, idx):
        return self.messages[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_del = self.get_batch_del(idx)
        batch_add = self.get_batch_add(idx)
        batch_y = self.get_batch_labels(idx)
        batch_messages = self.get_batch_messages(idx)
        return batch_texts, batch_del, batch_add, batch_messages, batch_y

# Multi-class Classification Model
class MultiClassifier(nn.Module):
    def __init__(self):
        super(MultiClassifier, self).__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained("mycodebert")
        self.bert2 = BertModel.from_pretrained("mybert")
        dropout_rate = 0.3
        self.patch_del_hidden = nn.Linear(768, 384)
        self.patch_add_hidden = nn.Linear(768, 384)
        self.hidden = nn.Sequential(
            nn.Linear(768 * 5, 768 * 3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768 * 3, 768),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.att1 = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        self.att2 = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)

    def forward(self, input_id, del_id, add_id, message_id, mask,
                del_mask, add_mask, message_mask):
        message = self.bert2(input_ids=message_id, attention_mask=message_mask, return_dict=False)[0]  # [20, 256, 768]
        add_patch = self.bert1.roberta(input_ids=add_id, attention_mask=add_mask, return_dict=False)[
            0]  # [20, 256, 768]
        del_patch = self.bert1.roberta(input_ids=del_id, attention_mask=del_mask, return_dict=False)[
            0]  # [20, 256, 768]
        text_output = self.bert2(input_ids=input_id, attention_mask=mask, return_dict=False)[0]  # [20, 512, 768]

        text = text_output[:, 1:, :]   # [20, 511, 768]
        text_cls = text_output[:, 0, :]
        message_cls = message[:, 0, :]   # [20, 768]

        patch = torch.cat((add_patch[:, 1:, :], del_patch[:, 1:, :]), dim=1)
        patch_avg = (add_patch[:, 0, :] + del_patch[:, 0, :]) / 2   # [20, 768]

        message = message.repeat(1, 2, 1)
        att_output_1 = self.att1(message, patch, patch)  # [20, 512, 768]
        att_avg_1 = torch.mean(att_output_1[0], dim=1)  # [20, 768]

        res_output_1 = patch_avg + att_avg_1  # [20, 768]

        input_1 = res_output_1

        att_output_2 = self.att2(text, patch, patch)  # [20, 511, 768]
        att_avg_2 = torch.mean(att_output_2[0], dim=1)  # [20, 768]
        res_output_2 = patch_avg + att_avg_2  # [20, 768]

        input_2 = res_output_2

        input = torch.cat((input_1, input_2, text_cls, message_cls, patch_avg), dim=1)  # [20 ,768 * 4]

        hidden_output = self.hidden(input)

        return hidden_output


# Train Multi-Class Classification Model
def train_multi_model(model, train_data, val_data, learning_rate, epochs):
    train_dataset = MultiClassDataset(dataframe=train_data)
    val_dataset = MultiClassDataset(dataframe=val_data)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=20, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=20)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    best_f1_score = 0.0
    print("Now loading the best multi-class model parameters...")
    try:
        with open(best_f1_path, 'r') as f:
            best_f1_score = float(f.read())
            f.close()
        model.load_state_dict(torch.load(best_model_path))
        print("Read best multi-class model successfully. Starting training...")
    except:
        print("Best multi-class model not found. Starting training from scratch...")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(epochs):
        model.train()
        total_loss_train = 0
        total_acc_train = 0

        for inputs, delete, add, messages, labels in tqdm(train_dataloader):
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            del_ids = delete['input_ids'].squeeze(1).to(device)
            add_ids = add['input_ids'].squeeze(1).to(device)
            message_ids = messages['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            del_mask = delete['attention_mask'].squeeze(1).to(device)
            add_mask = add['attention_mask'].squeeze(1).to(device)
            message_mask = messages['attention_mask'].squeeze(1).to(device)
            labels = labels.long().to(device)

            outputs = model(input_ids, del_ids, add_ids, message_ids, attention_mask, del_mask, add_mask, message_mask)
            preds = torch.argmax(outputs, dim=1)
            print(preds)

            loss = criterion(outputs, labels)
            total_loss_train += loss.item()

            acc = (preds == labels).sum().item()
            total_acc_train += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f'''Epochs: {epoch + 1}
                | Train Loss: {loss / len(labels): .3f}
                | Train Accuracy: {acc / len(labels): .2%} ''')

        model.eval()
        all_preds = []
        all_labels = []
        total_loss_val = 0
        total_acc_val = 0

        with torch.no_grad():
            for inputs, delete, add, messages, labels in tqdm(val_dataloader):
                input_ids = inputs['input_ids'].squeeze(1).to(device)
                del_ids = delete['input_ids'].squeeze(1).to(device)
                add_ids = add['input_ids'].squeeze(1).to(device)
                message_ids = messages['input_ids'].squeeze(1).to(device)
                attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                del_mask = delete['attention_mask'].squeeze(1).to(device)
                add_mask = add['attention_mask'].squeeze(1).to(device)
                message_mask = messages['attention_mask'].squeeze(1).to(device)
                labels = labels.long().to(device)

                outputs = model(input_ids, del_ids, add_ids, message_ids, attention_mask, del_mask, add_mask, message_mask)
                preds = torch.argmax(outputs, dim=1)

                loss = criterion(outputs, labels)

                total_loss_val += loss.item()
                total_acc_val += (preds == labels).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        for pred, label in zip(all_preds, all_labels):
            print(pred, label)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Epoch: {epoch + 1}")
        print(
            f"Train Loss: {total_loss_train / len(train_dataset):.3f} | Train Accuracy: {total_acc_train / len(train_dataset):.3f}")
        print(
            f"Val Loss: {total_loss_val / len(val_dataset):.3f} | Val Accuracy: {total_acc_val / len(val_dataset):.3f}")
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1 Score: {f1:.3f}")
        print()

        with open("out/result.txt", 'a') as f:
            f.write(f"Epoch: {epoch + 1}\n")
            f.write(
                f"Train Loss: {total_loss_train / len(train_dataset):.3f} | Train Accuracy: {total_acc_train / len(train_dataset):.3f}\n")
            f.write(
                f"Val Loss: {total_loss_val / len(val_dataset):.3f} | Val Accuracy: {total_acc_val / len(val_dataset):.3f}\n")
            f.write(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1 Score: {f1:.3f}\n\n")

        if f1 > best_f1_score:
            print("F1 score improved. Now saving the multi-class model...")
            print("\n\n")
            torch.save(model.state_dict(), best_model_path)
            best_f1_score = f1
            with open(best_f1_path, 'w') as f:
                f.write(str(best_f1_score))
                f.close()


# Train the Multi-Class Classification Model
NUM_CLASSES_MULTI = 8
multi_model = MultiClassifier()
LR_MULTI = 5e-5
EPOCHS_MULTI = 10
filtered_df_train = df_train[df_train['flag'] == 'positive']
filtered_df_test = df_test[df_test['flag'] == 'positive']
train_multi_model(multi_model, filtered_df_train, filtered_df_test, LR_MULTI, EPOCHS_MULTI)
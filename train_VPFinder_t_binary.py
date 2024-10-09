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
best_f1_path = "model_best/VPFinder_t_binary_f1.txt"
best_model_path = "model_best/VPFinder_t_binary_model.pth"

# Load data
df = pd.read_csv("data/dataset_240904.csv", low_memory=False)

# Split data into train and test sets
df_train, df_test = np.split(df, [int(.9 * len(df))])

# Load pre-trained tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("./mycodebert")
tokenizer2 = BertTokenizer.from_pretrained("./mybert")
labels = {'positive': 1, 'negative': 0}

class MyDataset(Dataset):
    def __init__(self, dataframe=df_train):
        self.texts = [tokenizer2(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for
                      text in dataframe['text']]
        self.labels = [labels[label] for label in dataframe['flag']]

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
        # Fetch a batch of texts
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

# Model
class VulClassifier(nn.Module):
    def __init__(self):
        super(VulClassifier, self).__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained("mycodebert")
        self.bert2 = BertModel.from_pretrained("mybert")
        self.dropout = 0.3
        self.hidden = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(8, 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )


    def forward(self, input_id, mask):
        pooled_output1 = self.bert2(input_ids=input_id, attention_mask=mask, return_dict=False)[0]
        output_text = pooled_output1[:, 0, :]
        # need new compare method
        output = self.hidden(output_text)
        return output

# Train
def train(model, train_data, val_data, learning_rate, epochs):
    train_dataset = MyDataset(dataframe=train_data)
    val_dataset = MyDataset(dataframe=val_data)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset),
                                                   batch_size=20, shuffle=False)   # [20, text, code, label]
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=20)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Now using {device}")

    best_f1_score = 0.0
    print("Now loading the best model parameters...")
    try:
        with open(best_f1_path, 'r') as f:
            best_f1_score = float(f.read())
            f.close()
        model.load_state_dict(torch.load(best_model_path))
        print("Read best model successfully. Starting training...")
    except:
        print("Best model not found. Starting training from scratch...")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if use_cuda:
        print("with cuda...")
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(epochs):
        model.train()
        total_loss_train = 0
        total_acc_train = 0

        for texts, labels in tqdm(train_dataloader):
            input_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].squeeze(1).to(device)
            labels = labels.long().to(device)

            outputs = model(input_ids, attention_mask)
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
            for texts, labels in tqdm(val_dataloader):
                input_ids = texts['input_ids'].squeeze(1).to(device)
                attention_mask = texts['attention_mask'].squeeze(1).to(device)
                labels = labels.long().to(device)

                outputs = model(input_ids, attention_mask)
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
            print("F1 score improved. Now saving the model...")
            print("\n\n")
            torch.save(model.state_dict(), best_model_path)
            best_f1_score = f1
            with open(best_f1_path, 'w') as f:
                f.write(str(best_f1_score))
                f.close()


EPOCHS = 2
model = VulClassifier()
LR = 5e-5
train(model, df_train, df_test, LR, EPOCHS)
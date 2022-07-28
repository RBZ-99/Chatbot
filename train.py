import json
from tkinter.ttk import LabeledScale
from nltk_utils import *
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))


ignore_words = ['?','!','.',','] #text and symbols to ignore
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


x_train = []
y_train = []
for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)

    label = tags.index(tag) #gives the index of the labels
    y_train.append(label)  #Not making it one hot encoding as we use cross_entropy loss

x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self,idx):
        return self.x_data[idx],self.y_data[idx]

    def __len__(self):
        return self.n_samples

batch_size = 8
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset,batch_size = batch_size,shuffle = True,num_workers = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size,hidden_size,output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr= learning_rate)
for e in range(epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model.forward(words)
        loss = criterion(outputs,labels)

        optimizer.zero_grad() #empty the gradients first
        loss.backward()
        optimizer.step()

    if (e+1)%100 == 0:
        print(f'epoch {e+1}/epochs, loss = {loss.item():.4f}')

data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words" : all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data,FILE)

print(f'Training complete. FIle save to {FILE}')
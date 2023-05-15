import pickle
import numpy as np

import torch
import torch.nn as nn


class Discriminator(torch.nn.Module):
    """
    Discriminator D
    """
    def __init__(self, voc_size, emb_size, batch_size, classes, hidden_size):
        super(Discriminator, self).__init__()
        #self.embedding = nn.Embedding(input_size, emb_size)
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.classes = classes
        self.hidden_size = hidden_size
        self.fc = nn.Linear(self.hidden_size, self.classes) # batch_size input_input # klassenazahl #self hidden size, classes 21.11
        self.softmax = nn.Softmax(dim=-1) #dim=1

    def forward(self, x):
        #x = x.long()
        x = x.float()
        #x = x.view(self.emb_size, self.input_size)
        #y = torch.squeeze(x).transpose(1,0)
        y = x #[0]
        #print(y)
        output = self.fc(y)
        output = self.softmax(output)
        output = output.squeeze()
        return output


# Generator G
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, emb_size, voc_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.voc_size = voc_size
        self.dropout = nn.Dropout(0.3)
        self.embedding = nn.Embedding(self.voc_size + 1, self.input_size, padding_idx=0)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers) #(self.input_size, self.hidden_size, self.num_layers)

    def forward(self, input):
        emb = self.embedding(input)
        embed = emb.view(input.shape[0], 1, -1) #-1, 1, self.input_size)
        output, hidden = self.lstm(embed)
        return output, hidden


class Comb(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, emb_size, voc_size, classes):
        super(Comb, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb_size = emb_size
        self.voc_size = voc_size
        self.classes = classes
        self.embedding = nn.Embedding(self.voc_size + 1, self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers) #(self.input_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.classes) # batch_size input_input # klassenazahl #self hidden size, classes 21.11
        self.softmax = nn.Softmax(dim=-1) #-1) #dim=1

    def forward(self, input):
        emb = self.embedding(input)
        embed = emb.view(input.shape[0], 1, -1) #-1, 1, self.input_size)
        output, hidden = self.lstm(embed)
        output = self.fc(output)
        output = self.softmax(output)
        output = output.squeeze()
        return output


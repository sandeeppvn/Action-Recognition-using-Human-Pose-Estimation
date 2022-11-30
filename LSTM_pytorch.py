import torch.nn as nn
import torch
from torch.utils.data import Dataset

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.batchNorm = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

        self.fc2 = nn.Linear(84480, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Set initial hidden and cell states
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # # Apply batch normalization
        # x = self.batchNorm(x)

        # Forward propagate LSTM
        # out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])

        #Flatten x and send it to fully connected layer
        x = x.flatten()
        x = self.fc2(x)

        #Apply softmax
        x = self.softmax(x)

        return x

class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        sequence = self.X[index]
        label = self.y[index]
        
        # Convert to tensor
        sequence = torch.tensor(sequence)
        label = torch.tensor(label)

        # Convert sequence to double
        sequence = sequence.double()
        return sequence, label
    
    def __len__(self):
        return self.X.shape[0]

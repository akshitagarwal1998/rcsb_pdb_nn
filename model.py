import torch
import torch.nn as nn

class ProteinClassifier(nn.Module):
    def __init__(self, hidden_dim=None, input_dim=None, dropout_rate=0.5):
        super(ProteinClassifier, self).__init__()
        
        input_dim = 3922 if input_dim is None else input_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        if hidden_dim is None:
            # Logistic regression (no hidden layer)
            self.linear = nn.Linear(input_dim, 1)
        else:
            # Fully connected network with BatchNorm and Dropout
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.hidden_dim is None:
            return self.linear(x)
        else:
            x = self.linear1(x)
            x = self.batchnorm1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x

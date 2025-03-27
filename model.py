import torch
import torch.nn as nn

class ProteinClassifier(nn.Module):
    def __init__(self, hidden_dim=None, input_dim=None):
        super(ProteinClassifier, self).__init__()
        
        input_dim = 3922 if input_dim is None else input_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if hidden_dim is None:
            # Logistic regression (no hidden layer)
            self.linear = nn.Linear(input_dim, 1)
        else:
            # Fully connected network
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.hidden_dim is None:
            return self.linear(x)
        else:
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x

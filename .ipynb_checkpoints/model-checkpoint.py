import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinClassifier(nn.Module):
    def __init__(self, hidden_dim=None):
        super(ProteinClassifier, self).__init__()

        self.hidden_dim = hidden_dim

        if hidden_dim is None:
            # Logistic regression (no hidden layer)
            self.linear = nn.Linear(2, 1)
        else:
            # Fully connected model: input -> hidden -> output
            self.fc1 = nn.Linear(2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.hidden_dim is None:
            out = self.linear(x)
        else:
            x = F.relu(self.fc1(x))
            out = self.fc2(x)
        return out  # Note: No sigmoid here — use BCEWithLogitsLoss

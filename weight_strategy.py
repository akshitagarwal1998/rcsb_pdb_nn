import torch

def inverse_class_weighting(dataset):
    """
    Assigns inverse class frequency weights to balance class distribution.
    Expects a dataset where each item returns (feature, label).
    """
    labels = torch.tensor([label.item() for _, label in dataset])
    class_counts = torch.tensor([(labels == t).sum() for t in torch.unique(labels)])
    weights = 1. / class_counts.float()
    return weights[labels.long()]

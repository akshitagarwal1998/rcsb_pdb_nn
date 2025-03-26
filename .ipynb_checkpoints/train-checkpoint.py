from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from model import ProteinClassifier
from dataset import ProteinPairDataset
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

def compute_sample_weights(dataset):
    labels = torch.tensor([label.item() for _, label in dataset])
    class_counts = torch.tensor([(labels == t).sum() for t in torch.unique(labels)])
    weights = 1. / class_counts.float()
    return weights[labels.long()]

def evaluate(model, dataloader, loss_fn):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x).squeeze()
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item()

            all_logits.extend(logits.detach().numpy())
            all_labels.extend(batch_y.numpy())

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    roc_auc = roc_auc_score(all_labels, probs)
    pr_auc = average_precision_score(all_labels, probs)
    mcc = matthews_corrcoef(all_labels, [int(p > 0.5) for p in probs])

    return total_loss, roc_auc, pr_auc, mcc

def train_model(
    cath_df=None,
    features=None,
    labels=None,
    hidden_dim=None,
    num_epochs=5,
    batch_size=4,
    val_split=0.2
):
    """
    Trains a model using either:
      - raw DataFrame (cath_df)
      - OR precomputed (features + labels)
    """

    if features and labels:
        dataset = ProteinPairDataset(features=features, labels=labels)
        dataset_size = len(dataset)
        split = int(val_split * dataset_size)
        indices = list(range(dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
    elif cath_df is not None:
        train_df, val_df = train_test_split(cath_df, test_size=val_split, random_state=42, shuffle=True)
        train_dataset = ProteinPairDataset(df=train_df)
        val_dataset = ProteinPairDataset(df=val_df)
    else:
        raise ValueError("Must pass either cath_df or features + labels")

    # Weighted sampler (train only)
    train_weights = compute_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model and optimizer
    model = ProteinClassifier(hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"Training model (hidden_dim={hidden_dim}) for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_logits, all_labels = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_x, batch_y in progress_bar:
            logits = model(batch_x).squeeze()
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_logits.extend(logits.detach().numpy())
            all_labels.extend(batch_y.numpy())

            progress_bar.set_postfix(loss=loss.item())

        # Eval
        val_loss, roc_auc, pr_auc, mcc = evaluate(model, val_loader, loss_fn)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"ROC AUC: {roc_auc:.3f} | PR AUC: {pr_auc:.3f} | MCC: {mcc:.3f}")

    return model


def test_model_on_ecod(model, ecod_df, batch_size=4):
    test_dataset = ProteinPairDataset(ecod_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss()
    test_loss, roc_auc, pr_auc, mcc = evaluate(model, test_loader, loss_fn)

    print(f"[FINAL TEST on ECOD] Loss: {test_loss:.4f} | ROC AUC: {roc_auc:.3f} | PR AUC: {pr_auc:.3f} | MCC: {mcc:.3f}")


if __name__ == "__main__":
    cath_df = pd.read_csv("./data/cath_moments.tsv", sep='\t', header=None).dropna(axis=1)
    ecod_df = pd.read_csv("./data/ecod_moments.tsv", sep='\t', header=None).dropna(axis=1)

    model = train_model(cath_df, hidden_dim=None, num_epochs=5)
    test_model_on_ecod(model, ecod_df)


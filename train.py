import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from tqdm.notebook import tqdm

from model import ProteinClassifier
from dataset import ProteinPairDataset,StreamingProteinPairDataset
from weight_strategy import inverse_class_weighting
from dataset import create_dataloaders

# Limit CPU usage to 80% of available cores
num_threads = max(1, int(os.cpu_count() * 0.8))
torch.set_num_threads(num_threads)
print(f"[INFO] Using {num_threads} CPU threads")

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x).squeeze()
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item()

            all_logits.extend(logits.detach().cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    roc_auc = roc_auc_score(all_labels, probs)
    pr_auc = average_precision_score(all_labels, probs)
    mcc = matthews_corrcoef(all_labels, [int(p > 0.5) for p in probs])

    return total_loss, roc_auc, pr_auc, mcc

def train_model(
    protein_df=None,
    features=None,
    labels=None,
    hidden_dim=None,
    input_dim=None,
    num_epochs=5,
    batch_size=64,
    val_split=0.2,
    writer=None,
    streaming=True 
):

    """
    Trains a model using either:
      - raw DataFrame (protein_df)
      - OR precomputed (features + labels)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if features and labels:
        dataset = ProteinPairDataset(features=features, labels=labels)
        dataset_size = len(dataset)
        split = int(val_split * dataset_size)
        indices = list(range(dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
    elif protein_df is not None:
        # Datasets handled inside create_dataloaders()
        pass

    else:
        raise ValueError("Must pass either protein_df or features + labels")
    
    print(f"[INFO] Loading Dataloader using streaming :",streaming)

    train_loader, val_loader = create_dataloaders(
        protein_df=protein_df,
        features=features,
        labels=labels,
        batch_size=batch_size,
        val_split=val_split,
        streaming=streaming
    )

    # Model and optimizer
    model = ProteinClassifier(hidden_dim=hidden_dim, input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"Training model (hidden_dim={hidden_dim}) for {num_epochs} epochs...")

    log_file = open("train_log.txt", "a")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_logits, all_labels = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x).squeeze()
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_logits.extend(logits.detach().cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        # Evaluation
        val_loss, roc_auc, pr_auc, mcc = evaluate(model, val_loader, loss_fn, device)

        msg = (f"Epoch {epoch+1}/{num_epochs} | "
               f"Train Loss: {train_loss:.4f} | "
               f"Val Loss: {val_loss:.4f} | "
               f"ROC AUC: {roc_auc:.3f} | PR AUC: {pr_auc:.3f} | MCC: {mcc:.3f}")

        print(msg)
        from datetime import datetime
        log_file.write(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {msg}\n")
        log_file.flush()
        os.fsync(log_file.fileno())

        if writer:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Metrics/ROC_AUC", roc_auc, epoch)
            writer.add_scalar("Metrics/PR_AUC", pr_auc, epoch)
            writer.add_scalar("Metrics/MCC", mcc, epoch)

    log_file.close()
    return model


def test_model_on_ecod(model, ecod_df, writer=None, batch_size=64, device="cpu", log_path="test_log.txt"):
    test_dataset = ProteinPairDataset(df=ecod_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss()
    test_loss, roc_auc, pr_auc, mcc = evaluate(model, test_loader, loss_fn, device)

    msg = (f"[FINAL TEST on ECOD] Loss: {test_loss:.4f} | ROC AUC: {roc_auc:.3f} | "
           f"PR AUC: {pr_auc:.3f} | MCC: {mcc:.3f}")
    print(msg)

    from datetime import datetime
    with open(log_path, "a") as f:
        f.write(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {msg}\n")

    if writer:
        writer.add_scalar("Test/Loss", test_loss)
        writer.add_scalar("Test/ROC_AUC", roc_auc)
        writer.add_scalar("Test/PR_AUC", pr_auc)
        writer.add_scalar("Test/MCC", mcc)

    return test_loss, roc_auc, pr_auc, mcc
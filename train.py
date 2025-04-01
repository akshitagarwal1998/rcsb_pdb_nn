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
from datetime import datetime
import shutil
timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

from model import ProteinClassifier
from dataset import ProteinPairDataset, StreamingProteinPairDatasetV2
from dataset import create_dataloaders
from tensorboard_utils import get_tensorboard_writer

num_threads = max(1, int(os.cpu_count()))
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
    protein_df,
    hidden_dim=None,
    input_dim=None,
    num_epochs=5,
    batch_size=64,
    val_split=0.2,
    writer=None,
    lr=1e-3
):

    """
    Trains a model using either:
      - raw DataFrame (protein_df)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    writer = get_tensorboard_writer(
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        val_split=val_split,
        tag="baseline"
    )

    # Extract log directory name from tensorboard writer for consistent logging
    log_dir_name = os.path.basename(writer.log_dir.rstrip('/'))
    os.makedirs("logs", exist_ok=True)
    log_file_path = os.path.join("logs", f"{log_dir_name}.txt")
    log_file = open(log_file_path, "a")
    os.makedirs("modelData", exist_ok=True)
    best_model_path = os.path.join("modelData", f"{log_dir_name}_best.pt")
    all_models_dir = os.path.join("modelData", f"{log_dir_name}_all")
    os.makedirs(all_models_dir, exist_ok=True)

    print(f"[INFO] Loading Dataloader")

    train_loader, val_loader = create_dataloaders(
        protein_df=protein_df,
        batch_size=batch_size,
        val_split=val_split
    )

    # Model and optimizer
    model = ProteinClassifier(hidden_dim=hidden_dim, input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Weight decay is L2
    loss_fn = nn.BCEWithLogitsLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7, verbose=True
    )

    print(f"Training model (hidden_dim={hidden_dim}) for {num_epochs} epochs...")

    best_roc_auc = 0
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

        epoch_model_path = os.path.join(all_models_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_model_path)

        if epoch == 0 or roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            torch.save(model.state_dict(), best_model_path)

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

        best_model_path = os.path.join("modelData", f"{log_dir_name}_best.pt")
        torch.save(model.state_dict(), best_model_path)

        # Save full checkpoint including optimizer for resume
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss
        }
        full_ckpt_path = os.path.join(all_models_dir, f"epoch_{epoch+1}_full.pt")
        torch.save(checkpoint, full_ckpt_path)

        scheduler.step(val_loss)

    writer.close()
    log_file.close()
    return model

def test_model_on_ecod(model, protein_df, writer=None, batch_size=64, device="cpu", log_path="test_log.txt"):
    test_dataset = StreamingProteinPairDatasetV2(protein_df)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_fn = nn.BCEWithLogitsLoss()
    test_loss, roc_auc, pr_auc, mcc = evaluate(model, test_loader, loss_fn, device)

    msg = (f"[FINAL TEST on ECOD] Loss: {test_loss:.4f} | ROC AUC: {roc_auc:.3f} | "
           f"PR AUC: {pr_auc:.3f} | MCC: {mcc:.3f}")
    print(msg)

    # Write to file
    with open(log_path, "a") as f:
        f.write(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {msg}\n")

    # TensorBoard
    if writer:
        writer.add_scalar("Test/Loss", test_loss)
        writer.add_scalar("Test/ROC_AUC", roc_auc)
        writer.add_scalar("Test/PR_AUC", pr_auc)
        writer.add_scalar("Test/MCC", mcc)

    return test_loss, roc_auc, pr_auc, mcc
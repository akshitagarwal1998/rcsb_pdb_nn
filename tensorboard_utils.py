from torch.utils.tensorboard import SummaryWriter
import time
import os

def get_tensorboard_writer(
    log_root="tensorboard_logs",
    run_name=None,
    hidden_dim=None,
    batch_size=None,
    lr=None,
    num_epochs=None,
    val_split=None,
    streaming=False,
    tag=None
):
    """
    Creates a SummaryWriter with a descriptive, versioned log directory.

    Args:
        log_root (str): Root folder for logs
        run_name (str): Custom name override
        hidden_dim (int): Hidden layer size (optional)
        batch_size (int): Batch size used
        lr (float): Learning rate
        num_epochs (int): Number of epochs
        tag (str): Optional tag like "baseline", "exp1", etc.

    Returns:
        SummaryWriter instance
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if run_name is None:
        parts = []
        if tag:
            parts.append(tag)
        if hidden_dim is not None:
            parts.append(f"h{hidden_dim}")
        if batch_size:
            parts.append(f"bs{batch_size}")
        if lr:
            parts.append(f"lr{lr}")
        if num_epochs:
            parts.append(f"ep{num_epochs}")
        parts.append(timestamp)
        run_name = "_".join(parts)

    log_dir = os.path.join(log_root, run_name)
    print(f"[TensorBoard] Logging to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)

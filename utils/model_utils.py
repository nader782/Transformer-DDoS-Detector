"""
model_utils.py
--------------
Transformer-based classifier for DDoS detection.

Architecture
------------
  Input (batch, seq_len, n_features)
      └─> Linear projection → d_model
      └─> Positional encoding (sinusoidal)
      └─> TransformerEncoder (n_layers × n_heads)
      └─> Global average pooling over seq dimension
      └─> Dropout
      └─> Linear head → n_classes
      └─> Softmax (during inference) / log-softmax (during training)
"""

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import joblib


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 512,
) -> tuple:
    """Wrap numpy arrays into PyTorch DataLoaders.

    Parameters
    ----------
    X_train, X_test : np.ndarray  – shape (n, n_features)
    y_train, y_test : np.ndarray  – shape (n, n_classes)  one-hot encoded
    batch_size : int

    Returns
    -------
    train_loader, test_loader : DataLoader
    """
    def _to_loader(X, y, shuffle):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    return _to_loader(X_train, y_train, True), _to_loader(X_test, y_test, False)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 10) -> tuple:
    """Apply a chronological sliding window to the dataset.
    
    The target label for the sequence is determined by majority vote.
    If there is a tie, the label of the last flow in the sequence is used.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, n_features)
    y : np.ndarray
        Shape (N, n_classes) for one-hot, or (N,) for integer labels.
    seq_len : int
        Number of flows per sequence.

    Returns
    -------
    X_seq, y_seq : np.ndarray
        X_seq shape: (N - seq_len + 1, seq_len, n_features)
        y_seq shape: (N - seq_len + 1, n_classes) or (N - seq_len + 1,)
    """
    X_seq = []
    y_seq = []
    
    # If y is one-hot encoded, convert to indices for easier majority voting
    y_is_onehot = (y.ndim == 2 and y.shape[1] > 1)
    if y_is_onehot:
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y

    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        
        # Get labels for this window
        window_labels = y_indices[i : i + seq_len]
        
        # Majority vote
        values, counts = np.unique(window_labels, return_counts=True)
        max_count = np.max(counts)
        majority_labels = values[counts == max_count]
        
        if len(majority_labels) == 1:
            seq_label = majority_labels[0]
        else:
            # Tie: take the last flow's label
            seq_label = window_labels[-1]
            
        if y_is_onehot:
            # Reconstruct one-hot
            onehot = np.zeros(y.shape[1], dtype=y.dtype)
            onehot[seq_label] = 1
            y_seq.append(onehot)
        else:
            y_seq.append(seq_label)
            
    return np.array(X_seq), np.array(y_seq)


# ---------------------------------------------------------------------------
# Positional encoding  (Vaswani et al., "Attention Is All You Need", 2017)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.

    Injects position information into the embedded token sequence so that
    the Transformer encoder can distinguish the temporal order of flows.
    Without this, self-attention is permutation-invariant and cannot learn
    order-dependent patterns.

    Parameters
    ----------
    d_model : int
        Embedding dimension (must match TransformerEncoder d_model).
    max_len : int
        Maximum sequence length the PE table supports.
    dropout : float
        Dropout applied after adding the positional signal.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)  – batch_first
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding then apply dropout.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """Transformer encoder for tabular / flow-level DDoS classification.

    Architecture
    ------------
      Input (batch, seq_len, n_features)
          └─> Linear projection → d_model
          └─> Positional encoding (sinusoidal)
          └─> TransformerEncoder (n_layers × n_heads)
          └─> Global average pooling over seq dimension
          └─> Dropout
          └─> Linear head → n_classes
          └─> Softmax (during inference) / log-softmax (during training)

    Parameters
    ----------
    n_features : int
        Number of input features (after selection).
    n_classes : int
        Number of output classes.
    d_model : int
        Transformer embedding dimension.
    n_heads : int
        Number of attention heads (must divide d_model evenly).
    n_layers : int
        Number of TransformerEncoder layers.
    dim_feedforward : int
        Hidden size of the feedforward sub-layer.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_model = d_model

        # Input projection: map raw features → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding: inject position information into each token
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(dim_feedforward // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, n_features)

        Returns
        -------
        logits : torch.Tensor  shape (batch, n_classes)
        """
        # Project to d_model
        x = self.input_proj(x)          # (batch, seq_len, d_model)
        # Add positional encoding
        x = self.pos_encoder(x)         # (batch, seq_len, d_model)
        # Transformer
        x = self.transformer(x)         # (batch, seq_len, d_model)
        
        # Aggregate sequence dimension using Global Average Pooling
        x = x.mean(dim=1)               # (batch, d_model)
        
        # Classify
        logits = self.classifier(x)     # (batch, n_classes)
        return logits


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: TransformerClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Run one training epoch.

    Returns
    -------
    avg_loss : float
    avg_acc  : float  (0–1)
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Support both 1-D integer labels and 2-D one-hot labels
        if y_batch.dim() == 2:
            targets = y_batch.argmax(dim=1)
        else:
            targets = y_batch.long()

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = X_batch.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        n += bs

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(
    model: TransformerClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Run one evaluation epoch.

    Returns
    -------
    avg_loss  : float
    avg_acc   : float  (0–1)
    all_preds : np.ndarray  int class indices
    all_true  : np.ndarray  int class indices
    all_probs : np.ndarray  softmax probabilities (n, n_classes)
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_true, all_probs = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Support both 1-D integer labels and 2-D one-hot labels
        if y_batch.dim() == 2:
            targets = y_batch.argmax(dim=1)
        else:
            targets = y_batch.long()

        logits = model(X_batch)
        loss = criterion(logits, targets)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        bs = X_batch.size(0)
        total_loss += loss.item() * bs
        correct += (preds == targets).sum().item()
        n += bs

        all_preds.append(preds.cpu().numpy())
        all_true.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    all_probs = np.concatenate(all_probs, axis=0)

    return total_loss / n, correct / n, all_preds, all_true, all_probs


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> dict:
    """Compute accuracy, per-class precision, recall, F1-score.

    Returns
    -------
    dict matching sklearn classification_report output_dict=True format,
    compatible with visualization helpers.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return report


def train_model(
    model: TransformerClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """Full training loop with per-epoch validation.

    Parameters
    ----------
    model         : TransformerClassifier
    train_loader  : DataLoader  (training split)
    test_loader   : DataLoader  (validation / test split)
    n_epochs      : int
    lr            : float        Initial learning rate
    device        : torch.device (auto-detect if None)
    verbose       : bool

    Returns
    -------
    history : dict
        Keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        Each value is a list of per-epoch values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs)

    # CrossEntropyLoss accepts integer class indices (1-D) directly.
    # train_epoch / eval_epoch convert 2-D one-hot targets to indices before
    # passing them to the criterion, so this works for both pipelines.
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _, _ = eval_epoch(
            model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if verbose:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:>3}/{n_epochs} │ "
                f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} │ "
                f"Val Loss: {va_loss:.4f}  Acc: {va_acc:.4f} │ "
                f"LR: {lr_now:.2e}"
            )

    return history


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    model: TransformerClassifier,
    scaler,
    label_encoder,
    selected_features: list,
    model_dir: str = "model",
) -> None:
    """Save the trained model and all required artifacts to *model_dir*.

    Files written
    -------------
    model/transformer_ddos.pt   – model weights + hyperparameters
    model/scaler.pkl             – fitted MinMaxScaler
    model/label_encoder.pkl      – fitted LabelBinarizer
    model/selected_features.json – list of selected feature names
    """
    os.makedirs(model_dir, exist_ok=True)

    # Model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_features": model.n_features,
        "n_classes":  model.n_classes,
        "d_model":    model.d_model,
        "n_heads":    model.transformer.layers[0].self_attn.num_heads,
        "n_layers":   len(model.transformer.layers),
        "dim_feedforward": model.transformer.layers[0].linear1.out_features,
        "dropout_rate":    model.transformer.layers[0].dropout.p,
    }
    torch.save(checkpoint, os.path.join(model_dir, "transformer_ddos.pt"))

    # Scaler & encoder
    joblib.dump(scaler,        os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

    # Feature names
    import json
    with open(os.path.join(model_dir, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    print(f"[model_utils] Model and artifacts saved to '{model_dir}/'")


def load_model(
    model_dir: str = "model",
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a previously saved TransformerClassifier and its metadata.

    Returns
    -------
    model          : TransformerClassifier (eval mode)
    scaler         : MinMaxScaler
    label_encoder  : LabelBinarizer
    selected_features : list[str]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        os.path.join(model_dir, "transformer_ddos.pt"),
        map_location=device,
    )

    model = TransformerClassifier(
        n_features=checkpoint["n_features"],
        n_classes=checkpoint["n_classes"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        dim_feedforward=checkpoint["dim_feedforward"],
        dropout=checkpoint["dropout_rate"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    import json
    with open(os.path.join(model_dir, "selected_features.json")) as f:
        selected_features = json.load(f)

    print(
        f"[model_utils] Model loaded from '{model_dir}/' on device: {device}")
    return model, scaler, label_encoder, selected_features

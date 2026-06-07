"""
LSTM model architectures for gold price prediction using PyTorch.
Includes standalone LSTM and LSTM-Attention combined model from research paper.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple


# ==========================================
# DEVICE DETECTION
# ==========================================
def get_device() -> torch.device:
    """
    Detect available device (CPU only).

    Returns:
        torch.device: CPU device
    """
    return torch.device("cpu")


# ==========================================
# MULTI-HEAD ATTENTION
# ==========================================
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        num_heads: Number of attention heads
        key_dim: Dimension of keys/queries
        output_dim: Output dimension (default: key_dim * num_heads)
    """

    def __init__(self, num_heads: int, key_dim: int, output_dim: int = None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.output_dim = output_dim or key_dim * num_heads

        self.W_q = nn.Linear(key_dim, self.output_dim)
        self.W_k = nn.Linear(key_dim, self.output_dim)
        self.W_v = nn.Linear(key_dim, self.output_dim)
        self.W_o = nn.Linear(self.output_dim, self.output_dim)

        self.scale = torch.sqrt(torch.FloatTensor([key_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, key_dim)

        Returns:
            Output tensor of shape (batch, seq_len, output_dim)
        """
        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, output_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        batch_size = Q.size(0)
        seq_len = Q.size(1)

        # Reshape for multi-head attention
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        head_dim = self.output_dim // self.num_heads
        Q = Q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)
        attention_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)

        # Concatenate heads
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.output_dim)
        )

        # Output projection
        output = self.W_o(attended)

        return output


# ==========================================
# MODEL ARCHITECTURES
# ==========================================
class StackedLSTM(nn.Module):
    """
    Stacked LSTM model for time series prediction.

    Architecture:
        LSTM(128) → Dropout(0.3) → BatchNorm →
        LSTM(64)  → Dropout(0.2) → BatchNorm →
        LSTM(32)  → Dropout(0.2) →
        Dense(32, ReLU) → Dense(1)
    """

    def __init__(
        self, input_size: int, hidden_sizes: list = None, dropout_rates: list = None
    ):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden sizes for each LSTM layer
            dropout_rates: List of dropout rates for each layer
        """
        super(StackedLSTM, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.2, 0.2]

        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        # Build LSTM layers
        for i, hidden_size in enumerate(hidden_sizes):
            input_dim = input_size if i == 0 else hidden_sizes[i - 1]
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_size, batch_first=True))
            self.batch_norm_layers.append(nn.BatchNorm1d(hidden_size))
            self.dropout_layers.append(nn.Dropout(dropout_rates[i]))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, 1)
        """
        # Pass through LSTM layers (full sequence)
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            x = dropout(x)

        # Take last timestep output
        x = x[:, -1, :]  # (batch, hidden_size)

        # Apply batch norm on the last layer output
        x = self.batch_norm_layers[-1](x)

        # Fully connected output
        out = self.fc(x)
        return out.squeeze(-1)


class LSTMAttentionModel(nn.Module):
    """
    LSTM-Attention Combined Model from research paper.

    Architecture (as per paper):
        LSTM(50 hidden units) →
        Multi-Head Attention (4 heads, key_dim=50) →
        Flatten →
        Dense(output)

    Supports direct multi-step forecasting via output_size > 1.
    This was the best performer in the paper (R²: 0.92).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 50,
        num_heads: int = 4,
        key_dim: int = 50,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden units (paper: 50)
            num_heads: Number of attention heads (paper: 4)
            key_dim: Dimension for attention keys (paper: 50)
            dropout: Dropout rate
            output_size: Number of future steps to predict (1 = single-step, >1 = direct multi-step)
        """
        super(LSTMAttentionModel, self).__init__()

        # Ensure hidden_size is divisible by num_heads
        # If not, adjust to nearest valid size
        if hidden_size % num_heads != 0:
            hidden_size = (hidden_size // num_heads) * num_heads
            if hidden_size == 0:
                hidden_size = num_heads

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # Multi-Head Self-Attention (PyTorch built-in)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        lstm_out = self.dropout1(lstm_out)

        # Multi-Head Self-Attention (residual connection + layer norm)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout2(attn_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # Residual connection

        # Take the output from the last time step
        out = attn_out[:, -1, :]  # (batch, hidden_size)

        # Dense output: (batch, output_size)
        out = self.fc(out)
        return out


class SimpleLSTM(nn.Module):
    """
    Simpler LSTM model for comparison or lighter training.
    LSTM(50) → Dropout(0.2) → Dense(1)
    """

    def __init__(self, input_size: int, hidden_size: int = 50, dropout: float = 0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(-1)


# ==========================================
# TRAINING UTILITIES
# ==========================================
def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    batch_size: int = 64,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience_es: int = 15,
    model_path: str = "models/best_lstm_model.pt",
) -> dict:
    """
    Train the LSTM model with early stopping and learning rate scheduling.

    Args:
        model: PyTorch model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        epochs: Maximum epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        patience_es: Early stopping patience
        model_path: Path to save best model

    Returns:
        Dictionary with training history (loss, val_loss, mae, val_mae)
    """
    device = get_device()
    model = model.to(device)

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    # Training history
    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    # Early stopping
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\nTraining for up to {epochs} epochs (batch_size={batch_size})...")
    print(f"Device: {device}")
    print(
        f"Train samples: {len(X_train)}, Val samples: {len(X_val) if X_val is not None else 0}"
    )

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n_train = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            train_mae += torch.mean(torch.abs(predictions - y_batch)).item() * len(
                X_batch
            )
            n_train += len(X_batch)

        avg_train_loss = train_loss / n_train
        avg_train_mae = train_mae / n_train
        history["loss"].append(avg_train_loss)
        history["mae"].append(avg_train_mae)

        # --- Validation phase ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            n_val = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)

                    val_loss += loss.item() * len(X_batch)
                    val_mae += torch.mean(
                        torch.abs(predictions - y_batch)
                    ).item() * len(X_batch)
                    n_val += len(X_batch)

            avg_val_loss = val_loss / n_val
            avg_val_mae = val_mae / n_val
            history["val_loss"].append(avg_val_loss)
            history["val_mae"].append(avg_val_mae)

            # Learning rate scheduler
            scheduler.step(avg_val_loss)

            # Print progress
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {avg_train_loss:.6f} - MAE: {avg_train_mae:.6f} - "
                f"Val Loss: {avg_val_loss:.6f} - Val MAE: {avg_val_mae:.6f} - "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0

                # Save best model
                _out_size = (
                    model.fc.out_features if hasattr(model.fc, "out_features") else 1
                )
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "input_size": X_train.shape[2],
                        "output_size": _out_size,
                        "hidden_sizes": [128, 64, 32],
                        "dropout_rates": [0.3, 0.2, 0.2],
                    },
                    model_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience_es:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {avg_train_loss:.6f} - MAE: {avg_train_mae:.6f}"
            )
            # No validation, just save every epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            _out_size = (
                model.fc.out_features if hasattr(model.fc, "out_features") else 1
            )
            torch.save(
                {
                    "model_state_dict": best_state,
                    "input_size": X_train.shape[2],
                    "output_size": _out_size,
                    "hidden_sizes": [128, 64, 32],
                    "dropout_rates": [0.3, 0.2, 0.2],
                },
                model_path,
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Training completed! Best val_loss: {best_val_loss:.6f}")
    print(f"Model saved to {model_path}")

    return history


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Make predictions with the trained model.

    Args:
        model: Trained PyTorch model
        X: Input features

    Returns:
        Predictions as numpy array
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions.cpu().numpy()


def load_model(model_path: str) -> nn.Module:
    """
    Load a saved model from checkpoint.

    Args:
        model_path: Path to saved model checkpoint

    Returns:
        Loaded PyTorch model
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = StackedLSTM(
        input_size=checkpoint["input_size"],
        hidden_sizes=checkpoint["hidden_sizes"],
        dropout_rates=checkpoint["dropout_rates"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    return model

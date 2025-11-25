import os
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from dataset import SignatureDataset
from model import SignatureNet
from sklearn.model_selection import train_test_split
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))  # project root if running from src/
DATA_DIR = os.path.join(ROOT, "data", "processed")
WEIGHTS_DIR = os.path.join(ROOT, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
MODEL_SAVE = os.path.join(WEIGHTS_DIR, "signature_model.pth")

BATCH_SIZE = 32
EPOCHS = 18
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------- Safe Weighted Sampler ----------
def get_weighted_sampler_from_labels(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
# ---------- Training Function ----------
def train():
    # Load full dataset
    full_dataset = SignatureDataset(root_dir=DATA_DIR, mode="train")
    num_samples = len(full_dataset)
    
    if num_samples == 0:
        raise ValueError(f"No images found in {DATA_DIR}. Check your dataset folders!")

    print(f"Full dataset size: {num_samples}")

    # Train/validation split
    indices = list(range(num_samples))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.12,
        random_state=42,
        stratify=full_dataset.labels
    )

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)

    print(f"Train samples: {len(train_set)}, Validation samples: {len(val_set)}")

    # Weighted sampler for training subset
    subset_labels = [full_dataset.labels[i] for i in train_idx]
    sampler = get_weighted_sampler_from_labels(subset_labels)

    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=min(BATCH_SIZE, len(train_set)),
        sampler=sampler,
        drop_last=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(BATCH_SIZE, len(val_set)),
        shuffle=False
    )

    # Model, loss, optimizer, scheduler
    model = SignatureNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        running_loss = 0.0
        for imgs, labs in train_loader:
            imgs = imgs.to(DEVICE)
            labs = labs.to(DEVICE).float()

            logits = model(imgs)
            loss = criterion(logits.squeeze(), labs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        tot = 0
        correct = 0

        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs = imgs.to(DEVICE)
                labs = labs.to(DEVICE).float()

                logits = model(imgs)
                loss = criterion(logits.squeeze(), labs)
                val_loss += loss.item() * imgs.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).long().squeeze()
                correct += (preds == labs.long()).sum().item()
                tot += labs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / tot if tot > 0 else 0.0

        print(f"Epoch {epoch}/{EPOCHS} | TrainLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | ValAcc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print("Saved best model:", MODEL_SAVE)

    print("Training complete. Best val acc:", best_val_acc)

if __name__ == "__main__":
    train()

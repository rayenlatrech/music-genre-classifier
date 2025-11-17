import os
import random
from typing import List, Tuple, Dict

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


# ------------ CONFIG ------------ #

DATA_DIR = os.path.join("data", "gtzan")  # same audio root as before
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_STATE = 42
N_MELS = 128
SAMPLE_RATE = 22050
DURATION = 30  # seconds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

GENRE_TO_IDX: Dict[str, int] = {g: i for i, g in enumerate(GENRES)}


# ------------ UTILITIES ------------ #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_audio_files(root_dir: str) -> List[Tuple[str, int]]:
    """
    Return list of (file_path, label_idx) for all audio files in GTZAN,
    skipping any files that cannot be opened by librosa.
    """
    raw_items = []
    for genre in GENRES:
        genre_dir = os.path.join(root_dir, genre)
        if not os.path.isdir(genre_dir):
            print(f"Warning: genre directory not found: {genre_dir}")
            continue

        for fname in os.listdir(genre_dir):
            if fname.lower().endswith((".wav", ".au", ".mp3")):
                fpath = os.path.join(genre_dir, fname)
                label_idx = GENRE_TO_IDX[genre]
                raw_items.append((fpath, label_idx))

    print(f"Found {len(raw_items)} audio files (before filtering).")

    valid_items = []
    for fpath, label_idx in raw_items:
        try:
            # try loading a short snippet just to verify it works
            y, sr = librosa.load(fpath, sr=SAMPLE_RATE, duration=1.0)
            if y is None or len(y) == 0:
                raise ValueError("Empty audio")
            valid_items.append((fpath, label_idx))
        except Exception as e:
            print(f"Skipping unreadable file: {fpath} | Reason: {e}")

    print(f"Kept {len(valid_items)} audio files after filtering.")
    return valid_items



def load_audio(file_path: str, duration: int = DURATION, sr: int = SAMPLE_RATE):
    """
    Load audio, ensuring consistent duration.
    If shorter, pad with zeros. If longer, center-crop.
    """
    y, sr = librosa.load(file_path, sr=sr)

    target_len = duration * sr

    if len(y) < target_len:
        # pad
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    elif len(y) > target_len:
        # center crop
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]

    return y, sr


def audio_to_melspec(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute log-mel spectrogram for given audio signal.
    Returns numpy array shape (1, H, W) normalized to [0, 1].
    """
    # Mel-spectrogram
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=N_MELS,
        power=2.0,
    )
    # Convert to log scale
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    # Normalize to [0, 1]
    melspec_norm = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-8)

    # Add channel dimension: (1, H, W)
    return melspec_norm.astype(np.float32)[np.newaxis, :, :]


# ------------ DATASET ------------ #

class GTZANSpectrogramDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], augment: bool = False):
        """
        items: list of (file_path, label_idx)
        augment: whether to apply simple data augmentation
        """
        self.items = items
        self.augment = augment

        # We'll use torchvision transforms on the spectrogram "image"
        # For now only random horizontal flip as a simple augmentation
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_path, label_idx = self.items[idx]

        y, sr = load_audio(file_path)
        melspec = audio_to_melspec(y, sr)  # (1, H, W)

        # Convert to tensor
        x = torch.from_numpy(melspec)  # (1, H, W)

        # For ResNet, we need 3 channels. Duplicate across channels.
        x = x.repeat(3, 1, 1)  # (3, H, W)

        if self.transform is not None:
            x = self.transform(x)

        y_tensor = torch.tensor(label_idx, dtype=torch.long)

        return x, y_tensor


# ------------ MODEL ------------ #

def build_model(num_classes: int = 10) -> nn.Module:
    """
    Use a ResNet18 pretrained on ImageNet, adapt final layer for our 10 genres.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ------------ TRAINING / EVAL ------------ #

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

        preds = outputs.argmax(dim=1)
        running_correct += (preds == batch_y).sum().item()
        running_total += batch_x.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item() * batch_x.size(0)

            preds = outputs.argmax(dim=1)
            running_correct += (preds == batch_y).sum().item()
            running_total += batch_x.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch_y.cpu().numpy().tolist())

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


def main():
    set_seed(RANDOM_STATE)
    print(f"Using device: {DEVICE}")

    # 1) List all audio files
    all_items = list_audio_files(DATA_DIR)

    # 2) Train/val/test split (e.g. 70/15/15)
    paths = [p for p, _ in all_items]
    labels = [lbl for _, lbl in all_items]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=RANDOM_STATE, stratify=labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_labels
    )

    def make_items(p_list, l_list):
        return list(zip(p_list, l_list))

    train_items = make_items(train_paths, train_labels)
    val_items = make_items(val_paths, val_labels)
    test_items = make_items(test_paths, test_labels)

    print(f"Train samples: {len(train_items)}")
    print(f"Val samples:   {len(val_items)}")
    print(f"Test samples:  {len(test_items)}")

    # 3) Create datasets & loaders
    train_dataset = GTZANSpectrogramDataset(train_items, augment=True)
    val_dataset = GTZANSpectrogramDataset(val_items, augment=False)
    test_dataset = GTZANSpectrogramDataset(test_items, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4) Build model
    model = build_model(num_classes=len(GENRES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_state_dict = None

    # 5) Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} "
            f"|| Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict().copy()
            print(f"--> New best val acc: {best_val_acc:.4f}")

    # Load best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 6) Final evaluation on test set
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)
    print("\n===== Final Test Performance =====")
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    # Confusion matrix & classification report (using sklearn)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=GENRES)
    import matplotlib.pyplot as plt

    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix - CNN (Mel-spectrogram)")
    plt.tight_layout()
    plt.show()

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=GENRES))

    # 7) Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "cnn_melspectrogram.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved CNN model to {model_path}")


if __name__ == "__main__":
    main()

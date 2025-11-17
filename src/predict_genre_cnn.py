import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn
from torchvision import models


# --------- CONFIG (must match train_cnn.py) --------- #

SAMPLE_RATE = 22050
DURATION = 30  # seconds
N_MELS = 128
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

MODEL_PATH = os.path.join("models", "music_genre_cnn.pt")


# --------- AUDIO + FEATURE FUNCTIONS --------- #

def load_audio(file_path: str, duration: int = DURATION, sr: int = SAMPLE_RATE):
    """
    Load audio and ensure fixed length (pad or center-crop).
    """
    y, sr = librosa.load(file_path, sr=sr)

    target_len = duration * sr
    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]

    return y, sr


def audio_to_melspec(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute normalized log-mel spectrogram: shape (1, H, W) in [0, 1].
    Must match train_cnn.py.
    """
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=N_MELS,
        power=2.0,
    )
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    melspec_norm = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-8)
    return melspec_norm.astype(np.float32)[np.newaxis, :, :]


# --------- MODEL DEFINITION (same as training) --------- #

def build_model(num_classes: int = 10) -> nn.Module:
    """
    ResNet18 with final layer adapted to 10 genres.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_trained_model(model_path: str = MODEL_PATH) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = build_model(num_classes=len(GENRES))
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# --------- PREDICTION FUNCTION --------- #

def predict_genre(audio_path: str, top_k: int = 3):
    """
    Load an audio file, run it through the CNN, and return
    top-k genres with probabilities.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")
    model = load_trained_model(MODEL_PATH)

    print(f"\nProcessing audio file: {audio_path}")
    y, sr = load_audio(audio_path)
    melspec = audio_to_melspec(y, sr)  # (1, H, W)
    x = torch.from_numpy(melspec).unsqueeze(0)  # (1, 1, H, W)
    x = x.repeat(1, 3, 1, 1)  # (1, 3, H, W) for ResNet

    x = x.to(DEVICE)

    with torch.inference_mode():
        outputs = model(x)  # (1, num_classes)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

    # Get top-k indices
    top_k = min(top_k, len(GENRES))
    top_indices = probs.argsort()[::-1][:top_k]
    top_genres = [GENRES[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]

    return list(zip(top_genres, top_probs))


# --------- CLI --------- #

def main():
    if len(sys.argv) < 2:
        print("Usage: python src\\predict_genre_cnn.py <path_to_audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    results = predict_genre(audio_path, top_k=3)

    print("\nTop predictions:")
    for genre, prob in results:
        print(f"  {genre:10s} : {prob:.3f}")


if __name__ == "__main__":
    main()

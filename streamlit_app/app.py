import os
import tempfile

import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
from torchvision import models

import matplotlib.pyplot as plt
import streamlit as st


# ---------- CONFIG (must match training) ---------- #

SAMPLE_RATE = 22050
DURATION = 30  # seconds
N_MELS = 128

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- AUDIO & FEATURES (same as training) ---------- #

def load_audio_fixed(path, duration=DURATION, sr=SAMPLE_RATE):
    """Load audio and pad/crop to fixed length."""
    y, sr = librosa.load(path, sr=sr)

    target_len = duration * sr
    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]

    return y, sr


def audio_to_melspec(y, sr):
    """Compute normalized log-mel spectrogram (1, H, W) in [0, 1]."""
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


# ---------- MODEL LOADING ---------- #

def build_model(num_classes=len(GENRES)):
    """ResNet18 with final layer adapted to our 10 genres."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@st.cache_resource
def load_model():
    """Load trained CNN once and cache it."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_top_k(y, sr, top_k=3):
    """Run CNN on raw audio and return top-k (genre, prob)."""
    model = load_model()

    melspec = audio_to_melspec(y, sr)  # (1, H, W)
    x = torch.from_numpy(melspec).unsqueeze(0)  # (1, 1, H, W)
    x = x.repeat(1, 3, 1, 1).to(DEVICE)        # (1, 3, H, W)

    with torch.inference_mode():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

    top_k = min(top_k, len(GENRES))
    top_indices = probs.argsort()[::-1][:top_k]
    top_genres = [GENRES[i] for i in top_indices]
    top_probs = [float(probs[i]) for i in top_indices]

    return list(zip(top_genres, top_probs)), probs


# ---------- STREAMLIT UI ---------- #

def main():
    st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")
    st.title("🎵 Music Genre Classifier (CNN on Mel-spectrograms)")
    st.markdown(
        """
        Upload an audio file (preferably ~30s, WAV/MP3), and the model will:
        - Play the audio
        - Show its mel-spectrogram
        - Predict the most likely genre using a CNN (ResNet18)
        """
    )

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "au", "ogg"])

    if uploaded_file is not None:
        # Save to a temporary file so librosa can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load & preprocess
        y, sr = load_audio_fixed(tmp_path)

        st.subheader("Audio preview")
        st.audio(tmp_path, format="audio/wav")

        # Predict
        with st.spinner("Analyzing and classifying..."):
            top3, all_probs = predict_top_k(y, sr, top_k=3)

        # Show predictions
        st.subheader("Top predictions")
        for genre, prob in top3:
            st.write(f"**{genre}**: {prob:.3f}")

        # Bar chart for all genres
        st.subheader("Prediction probabilities for all genres")
        st.bar_chart(
            {
                "genre": GENRES,
                "probability": all_probs,
            },
            x="genre",
            y="probability",
        )

                # Show mel-spectrogram (for visualization only)
        st.subheader("Mel-spectrogram")
        fig, ax = plt.subplots(figsize=(8, 4))

        # Compute a fresh mel-spectrogram in dB for display
        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=N_MELS,
            power=2.0,
        )
        melspec_db = librosa.power_to_db(melspec, ref=np.max)

        img = librosa.display.specshow(
            melspec_db,
            sr=sr,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            ax=ax,
        )
        ax.set_title("Mel-spectrogram (log scale)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)


        # Clean up temp file
        os.remove(tmp_path)

    else:
        st.info("Upload an audio file to get started.")


if __name__ == "__main__":
    main()

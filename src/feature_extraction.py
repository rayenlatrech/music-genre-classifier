import os
import warnings

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress some annoying librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------ CONFIG ------------ #

DATA_DIR = os.path.join("data", "gtzan")  # where your genre folders are
OUTPUT_CSV = os.path.join("data", "gtzan_features.csv")

# The 10 GTZAN genres (folder names)
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


def extract_features(file_path, duration=30):
    """
    Extract audio features from one file:
      - Tempo
      - Spectral centroid (mean, std)
      - Spectral bandwidth (mean, std)
      - Spectral rolloff (mean, std)
      - Zero crossing rate (mean, std)
      - MFCCs (13 coefficients: mean + std each)

    Returns:
        np.ndarray of shape (N_features,)
    """
    try:
        y, sr = librosa.load(file_path, duration=duration, mono=True)

        # If file is too short or empty
        if y is None or len(y) == 0:
            raise ValueError("Empty audio")

        # ---- Tempo ---- #
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # ---- Spectral Centroid ---- #
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_centroid_mean = np.mean(spec_centroid)
        spec_centroid_std = np.std(spec_centroid)

        # ---- Spectral Bandwidth ---- #
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bw_mean = np.mean(spec_bw)
        spec_bw_std = np.std(spec_bw)

        # ---- Spectral Rolloff ---- #
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)

        # ---- Zero Crossing Rate ---- #
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # ---- MFCCs ---- #
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_stds = np.std(mfcc, axis=1)

        # Concatenate all features into one vector
        features = np.hstack(
            [
                tempo,
                spec_centroid_mean,
                spec_centroid_std,
                spec_bw_mean,
                spec_bw_std,
                rolloff_mean,
                rolloff_std,
                zcr_mean,
                zcr_std,
                mfcc_means,
                mfcc_stds,
            ]
        )

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def build_dataset():
    """
    Loop over all audio files in data/gtzan/<genre>/,
    extract features, and build a pandas DataFrame.
    """
    rows = []
    feature_names = [
        "tempo",
        "spec_centroid_mean",
        "spec_centroid_std",
        "spec_bw_mean",
        "spec_bw_std",
        "rolloff_mean",
        "rolloff_std",
        "zcr_mean",
        "zcr_std",
    ]

    # Add MFCC names: mfcc1_mean ... mfcc13_mean, mfcc1_std ... mfcc13_std
    mfcc_mean_names = [f"mfcc_{i}_mean" for i in range(1, 14)]
    mfcc_std_names = [f"mfcc_{i}_std" for i in range(1, 14)]

    columns = feature_names + mfcc_mean_names + mfcc_std_names + ["genre", "filename"]

    for genre in GENRES:
        genre_dir = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_dir):
            print(f"Warning: directory not found for genre '{genre}': {genre_dir}")
            continue

        print(f"Processing genre: {genre}")
        files = [
            f for f in os.listdir(genre_dir)
            if f.lower().endswith((".wav", ".au", ".mp3"))
        ]

        for file in tqdm(files):
            file_path = os.path.join(genre_dir, file)
            features = extract_features(file_path)

            if features is None:
                continue

            row = list(features) + [genre, file]
            rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    return df


def main():
    print("Building dataset from audio files...")
    df = build_dataset()

    print("\nPreview of extracted features:")
    print(df.head())

    print(f"\nSaving features to: {OUTPUT_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone!")
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(df['genre'].value_counts())


if __name__ == "__main__":
    main()

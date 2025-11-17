# 🎵 Music Genre Classification  
### Classical ML (KNN / SVM / Random Forest) vs Deep Learning (CNN on Mel-Spectrograms)

This project explores music genre classification using the GTZAN dataset, comparing classical machine learning models with a deep-learning CNN trained on log-Mel spectrograms. It includes:

- Traditional ML models (KNN, SVM, RandomForest)
- Feature extraction (MFCCs and spectral features)
- Deep learning using a ResNet18 CNN
- Streamlit web application
- Prediction script for inference
- Full pipeline: extraction → training → evaluation → deployment

---

## 📂 Project Structure

```
music-genre-classifier/
│
├── data/
│   └── gtzan/
│
├── models/
│   ├── best_model.pkl
│   └── music_genre_cnn.pt
│
├── src/
│   ├── feature_extraction.py
│   ├── train_models.py
│   ├── train_cnn.py
│   └── predict_genre_cnn.py
│
└── streamlit_app/
    └── app.py
```

---

## 🎧 GTZAN Dataset

- 1000 audio tracks, 30 seconds each  
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

---

# 🧪 Classical Machine Learning Models

### Extracted Features:
- MFCC means (13)
- MFCC standard deviations (13)
- Spectral centroid (mean/std)
- Spectral bandwidth (mean/std)
- Spectral rolloff
- Zero crossing rate
- Tempo (BPM)

### Performance:

| Model | Accuracy | Macro-F1 | Notes |
|-------|----------|----------|-------|
| KNN | 0.595 | 0.598 | Struggles with MFCC dimensionality |
| **SVM (RBF)** | **0.685** | **0.686** | Best classical model |
| Random Forest | 0.565 | 0.560 | Underperforms on dense features |

---

# 🔥 Deep Learning — CNN on Mel-Spectrograms

Using ResNet18 pretrained on ImageNet and trained on 128-band log-Mel spectrograms.

### Model Details:
- ResNet18 backbone
- Adam optimizer
- Input: normalized log-Mel spectrogram
- 20 epochs

### CNN Performance:
- **Test Accuracy: 0.847**
- **Macro-F1: 0.84**

---

# 🌐 Streamlit Web App

Run:

```
streamlit run streamlit_app/app.py
```

Features:
- Upload audio
- Audio playback
- Mel-spectrogram visualization
- Top-3 predictions
- Probability chart

---

# 🧠 Prediction Script

Usage:

```
python src/predict_genre_cnn.py path/to/audio.wav
```

Output example:

```
rock : 0.812
metal: 0.103
blues: 0.045
```

---

# ⚙️ Training

Classical ML:

```
python src/train_models.py
```

Deep Learning:

```
python src/train_cnn.py
```

---

# 🛠 Technologies Used

Python, Librosa, NumPy, Pandas, Scikit-learn, PyTorch, TorchVision, Streamlit, Matplotlib

---

# 🚀 Key Learnings

- Classical ML provides useful baselines  
- CNN substantially outperforms classical methods  
- Spectrograms convert audio signals into image-like inputs  
- Full ML lifecycle implemented end-to-end  

---

# 📈 Future Improvements

- Data augmentation  
- Larger CNNs (ResNet34, EfficientNet)  
- Transformer-based audio models  
- Cloud deployment (Streamlit Cloud or HuggingFace Spaces)

---

# 📫 Contact

**Rayen Latrech**  
GitHub: https://github.com/rayenlatrech

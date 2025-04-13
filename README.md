
# 🎵 MelodyMind: Music Genre Predictor
MelodyMind is an AI-powered web app that classifies music genres using deep learning and audio signal processing. Upload a song clip, and the model analyzes it to predict one of 10 possible genres using Mel spectrograms and a trained CNN model.



## 🚀 Features

🎧 Genre Prediction: Upload MP3 files and get real-time genre predictions.

🧠 CNN Model: Built on TensorFlow/Keras and trained on the GTZAN dataset.

🎨 Stylish UI: Modern, dark-themed interface with emoji-enhanced genre results.

## 📊 Visualizations:

Prediction confidence bar chart

Mel spectrogram display of the uploaded audio

## 🎼 Supported Genres
Blues 🎷

Classical 🎻

Country 🤠

Disco 🪩

Hip-hop 🎤

Jazz 🎺

Metal 🤘

Pop 🎧

Reggae 🌴

Rock 🎸

## 📁 Dataset
Uses the GTZAN dataset:

10 genres

100 audio files per genre (30 seconds each)

Converted into Mel spectrograms for model input

## 🛠️ Tech Stack
Frontend: Streamlit

Backend: TensorFlow/Keras

Audio Processing: Librosa

Visualization: Seaborn, Matplotlib

## 🔧 Installation
bash
```
git clone https://github.com/yourusername/melodymind.git

pip install -r requirements.txt
```
Place your trained model file (genre_classifier.h5) and assets like logo.png in the root directory.

## ▶️ Run the App
bash
```
streamlit run app.py
```
# 📸 Screenshots

![image](https://github.com/user-attachments/assets/304327a4-c23e-4223-8d78-46b4990379e3)


![image](https://github.com/user-attachments/assets/3435ce89-011a-4444-af8c-a202fa65e7d0)

## 👨‍💻 Authors
Sundaram Dubey

## Contributions welcome!

📜 License
MIT License. See LICENSE for details.


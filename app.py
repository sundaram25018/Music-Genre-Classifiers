import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Genre labels
GENRE_LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("genre_classifier.h5")

# Preprocess audio
def load_and_preprocess_data(file_path, target_shape=(150, 150), show_progress=False):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    if show_progress:
        progress_bar = st.progress(0)
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        # Pad if too short
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram.numpy())

        if show_progress:
            progress_bar.progress((i + 1) / num_chunks)

    return np.array(data)

# Predict genre
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    average_prediction = np.mean(y_pred, axis=0)
    predicted_index = np.argmax(average_prediction)
    confidence = np.max(average_prediction)
    return predicted_index, confidence, average_prediction

# Plot spectrogram
def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    st.pyplot(plt)

# Plot confidence bar chart
def plot_confidence_chart(confidences):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=GENRE_LABELS, y=confidences, palette="viridis", ax=ax)
    ax.set_title("Model Confidence by Genre")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Apply custom styles
st.markdown("""
    <style>
        .stApp {
            background-color: #212529;
            color: #edede9;
        }
        h1, h2, h3 {
            color: #edede9;
        }
        .block-container {
            padding: 2rem;
        }
        .css-1cpxqw2 {
            background-color: #501537;
            border-radius: 12px;
            padding: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎛 Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Genre Classification"])

# Home Page
if app_mode == "Home":
    st.markdown("""
            <h1 style='color:#facc15'>MelodyMind: Genre Predictor</h1>
        </div>
    """, unsafe_allow_html=True)
    st.image("bg.jpg", use_column_width=True)
    st.markdown("""
        **Discover the power of AI in music analysis!**

        ### How It Works
        1. Upload Audio
        2. System analyzes audio and predicts genre
        3. See detailed results

        ### Why Choose Us?
        - 🎯 **Accurate** CNN model
        - ⚡ **Fast** predictions
        - 🧠 **Smart** chunk-based classification

        ### Try it now in the **Genre Classification** tab!
    """)

# About
elif app_mode == "About Project":
    st.markdown("""
    ### About the Project
    A CNN-based music genre classifier built on the GTZAN dataset.

    **Dataset:**
    - 10 genres (blues, classical, etc.)
    - 100 audio files per genre
    - Mel spectrograms used for input

    **Model:**
    - Trained with TensorFlow/Keras
    - Spectrograms resized to 150x150
    - Uses overlapping chunks for better prediction
    """)

# Prediction Page
elif app_mode == "🎧 Prediction":
    st.title("🎧 Upload Audio for Genre Prediction")

    audio_file = st.file_uploader("Upload an audio file (.mp3 or .wav)", type=["mp3", "wav"])
    
    if audio_file:
        st.audio(audio_file, format="audio/mp3")

        if not os.path.exists("Test_Music"):
            os.makedirs("Test_Music")
        file_path = os.path.join("Test_Music", audio_file.name)
        with open(file_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success("✅ File uploaded and saved!")

        if st.button("🔍 Analyze and Predict"):
            with st.spinner("Analyzing audio..."):
                X_test = load_and_preprocess_data(file_path, show_progress=True)
                predicted_index, confidence, all_probs = model_prediction(X_test)
                predicted_genre = GENRE_LABELS[predicted_index]

            st.balloons()
            st.success(f"🎵 Predicted Genre: **{predicted_genre.upper()}**")
            st.info(f"📈 Model Confidence: **{confidence:.2f}**")

            st.subheader("🎼 Mel Spectrogram of Uploaded Audio")
            plot_spectrogram(file_path)

            st.subheader("📊 Genre Confidence Chart")
            plot_confidence_chart(all_probs)

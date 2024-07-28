from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
predictions = []
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp3'}

app.config['UPLOAD_FOLDER'] = 'static/uploads'

Model_file = 'deepfake_final.h5'
model = load_model(Model_file)

# Load the scaler
scaler = joblib.load('scaler.save')

ml_model = joblib.load('audio_classifier_model.joblib')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def convert_to_wav(mp3_file):
    wav_file = os.path.join(UPLOAD_FOLDER, os.path.splitext(os.path.basename(mp3_file))[0] + '.wav')
    sound, sr = librosa.load(mp3_file, sr=None)
    wavfile.write(wav_file, sr, (sound * 32767).astype(np.int16))
    return wav_file

def convert_to_spectrogram(wav_file):
    y, sr = librosa.load(wav_file, sr=None)
    duration_sec = librosa.get_duration(y=y, sr=sr)
    if duration_sec > 60:
        num_segments = int(duration_sec / 60) + 1
        for i in range(num_segments):
            start_time = i * 60
            end_time = min((i + 1) * 60, duration_sec)
            segment = y[int(start_time * sr):int(end_time * sr)]
            spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
            plt.figure(figsize=(7.75, 3.85))  # Set figure size to 775x385
            librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sr, x_axis=None, y_axis=None)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust margins
            plt.axis('tight')  # Remove extra whitespace
            spectrogram_file = os.path.join(UPLOAD_FOLDER, os.path.splitext(os.path.basename(wav_file))[0] + f'_segment_{i}_spectrogram.png')
            plt.savefig(spectrogram_file, bbox_inches='tight', pad_inches=0)
            plt.close()
            img = image.load_img(spectrogram_file, target_size=(775, 385))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)[0]
            print(prediction[0])
            predictions.append(prediction)
    else:
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        plt.figure(figsize=(7.75, 3.85)) 
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sr, x_axis=None, y_axis=None)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
        plt.axis('tight')  
        spectrogram_file = os.path.join(UPLOAD_FOLDER, os.path.splitext(os.path.basename(wav_file))[0] + '_spectrogram.png')
        plt.savefig(spectrogram_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        img = image.load_img(spectrogram_file, target_size=(775, 385)) 
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predictions.append(prediction)
    return spectrogram_file.replace('\\', '/'), prediction

def delete_uploaded_files():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            

def extract_and_process_features(file_path):
    # Load the audio file
    data, sample_rate = librosa.load(file_path, sr=None)
    file_info = sf.info(file_path)

    # Basic Features
    duration = librosa.get_duration(y=data, sr=sample_rate)
    bit_depth = file_info.subtype_info
    channels = file_info.channels

    # Extracting various audio features
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=data)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    chroma_stft = librosa.feature.chroma_stft(y=data, sr=sample_rate)
    tonnetz = librosa.feature.tonnetz(y=data, sr=sample_rate)

    # Additional features can be computed here...

    # Averaging or summarizing the extracted features
    features = {
        'Duration': duration,
        'SamplingRate': sample_rate,
        'BitDepth': bit_depth,
        'Channels': channels,
        'SpectralCentroid': np.mean(spectral_centroid),
        'SpectralBandwidth': np.mean(spectral_bandwidth),
        'SpectralRolloff': np.mean(spectral_rolloff),
        'ZeroCrossingRate': np.mean(zero_crossing_rate),
        'MFCCs': np.mean(mfccs, axis=1),
        'ChromaFrequencies': np.mean(chroma_stft, axis=1),
        'Tonnetz': np.mean(tonnetz, axis=1)
    }

    # Create DataFrame from the extracted features
    df = pd.DataFrame([features])

    # Drop unnecessary columns
    df.drop(columns=['BitDepth', 'Channels'], inplace=True)

    # Convert string features to numeric
    df['MFCCs_mean'] = df['MFCCs'].apply(lambda x: np.mean(x))
    df['ChromaFrequencies_mean'] = df['ChromaFrequencies'].apply(lambda x: np.mean(x))
    df['Tonnetz_mean'] = df['Tonnetz'].apply(lambda x: np.mean(x))

    # Drop original string features
    df.drop(columns=['MFCCs', 'ChromaFrequencies', 'Tonnetz'], inplace=True)

    return df

def make_prediction(input_data):
    # Scale the input data
    scaled_data = scaler.transform(input_data)

    prediction = ml_model.predict_proba(scaled_data)
    percentage_real = prediction[0][0] * 100
    percentage_fake = 100 - percentage_real
    print("Percentage Fake",round(percentage_fake,2), "Percentage Real",percentage_real)
    return {"Percentage Fake":round(percentage_fake,2), "Percentage Real": percentage_real}


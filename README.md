# Neurocare
import pandas as pd
import librosa
import numpy as np
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from google.colab import files
uploaded = files.upload()
df=pd.read_csv('Parkinson_disease.csv')
# Drop non-numeric column
X = df.drop(columns=['name', 'status'])
y = df['status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy,report

from google.colab import files
uploaded_audio = files.upload() # Use a different variable name for audio upload
audio_file_name = list(uploaded_audio.keys())[0] # Get the actual uploaded file name

import librosa
y_audio, sr_audio = librosa.load(audio_file_name) # Load audio using the correct file name

def extract_voice_features(y,sr):
      print(audio_file_name) # Print the correct audio file name
      # librosa.pyin returns f0, voiced_flag, voiced_probs
      f0, voiced_flag, voiced_probs = librosa.pyin(
                       y, fmin=50, fmax=400, sr=sr )
      f0 = f0[~np.isnan(f0)]
      mean_F0 = np.mean(f0)
      max_F0 = np.max(f0)
      min_F0 = np.min(f0)

      jitter_percent = np.mean(np.abs(np.diff(f0))) / mean_F0 if mean_F0 != 0 else 0
      jitter_abs = np.mean(np.abs(np.diff(f0)))
      RAP = np.std(f0[:3]) if len(f0) > 3 else (np.std(f0) if len(f0) > 0 else 0)
      PPQ = np.std(f0[:5]) if len(f0) > 5 else (np.std(f0) if len(f0) > 0 else 0)
      DDP = jitter_abs * 3

      rms = librosa.feature.rms(y=y)[0]
      shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms) if np.mean(rms) != 0 else 0
      shimmer_db = 20 * np.log10(np.mean(rms)) if np.mean(rms) != 0 else 0
      APQ3 = np.std(rms[:3]) if len(rms) > 3 else (np.std(rms) if len(rms) > 0 else 0)
      APQ5 = np.std(rms[:5]) if len(rms) > 5 else (np.std(rms) if len(rms) > 0 else 0)
      APQ = np.std(rms)
      DDA = APQ3 * 3

      S = librosa.stft(y)
      harm, noise = librosa.decompose.hpss(S)
      harm_audio = librosa.istft(harm)
      noise_audio = librosa.istft(noise)
      HNR = np.mean(harm_audio**2) / (np.mean(noise_audio**2) + 1e-6)

      
      NHR = np.mean(noise_audio**2) / (np.mean(harm_audio**2) + 1e-6)
      RPDE = np.var(f0)
      DFA = np.mean(np.abs(np.diff(rms)))

     
      spread1 = np.mean(np.diff(f0)) if len(f0) > 1 else 0
      spread2 = np.var(np.diff(f0)) if len(f0) > 1 else 0
      D2 = np.std(f0)
      PPE = np.var(np.diff(f0)) / mean_F0 if mean_F0 != 0 else 0


      return np.array([ mean_F0, max_F0, min_F0, jitter_percent, jitter_abs, RAP, PPQ, DDP, shimmer,
                       shimmer_db, APQ3, APQ5, APQ,DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE ])


x = extract_voice_features(y_audio, sr_audio) # Pass y_audio and sr_audio
features= x.reshape(1, -1)
features_scaled = scaler.transform(features)



# Predict using your model
prediction = model.predict(features_scaled)[0] # Use features_scaled
probability = model.predict_proba(features_scaled)[0][1] # Use features_scaled
print(" The probability of getting Parkinson's is ", probability," and the prediction is ",prediction)

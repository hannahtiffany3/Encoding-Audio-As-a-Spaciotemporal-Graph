import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import scipy

def load_audio(file):
    file_path = f"../audio_signals/{file}"
    audio, fs = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono
    return audio, fs

def plot_audio(audio, fs):
    t = np.arange(len(audio)) / fs
    plt.figure()
    plt.plot(t, audio)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Signal")
    plt.show()

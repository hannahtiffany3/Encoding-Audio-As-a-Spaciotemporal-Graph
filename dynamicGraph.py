import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import scipy
import networkx as nx

eps = 1e-12

def load_audio(file):
    file_path = f"audio_signals/{file}"
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

def build_time_slice_graphs(Zxx, r=4, eps=1e-12):
    F, T = Zxx.shape
    graphs = []

    for k in range(T):
        #amplitudes for time slice k
        a = np.abs(Zxx[:, k]).astype(float)
        #normalize
        den = float(np.sum(a*a) + eps)

        G = nx.Graph()  # undirected local-frequency graph

        # nodes: frequency bins
        for i in range(F):
            # node id is the frequency-bin index i
            # store features as attributes
            log_power = 10.0 * np.log10(a[i]*a[i] + eps)
            G.add_node(i, amp=a[i], log_power=log_power)

        # edges: local neighborhood within +/- r
        # weight based on amplitude similarity (normalized dot product)
        for i in range(F):
            for d in range(1, r+1):
                j = i + d
                if j < F:
                    w = (a[i] * a[j]) / den
                    G.add_edge(i, j, weight=w)

        graphs.append(G)

    return graphs


audio_files = ["tone.wav"]
for file in audio_files:
    audio, fs = load_audio(file)
    #plot_audio(audio, fs)
    frame_ms = 25
    hop_ms = 10
    frame_len = int(round(frame_ms * 1e-3 * fs))
    hop_len = int(round(hop_ms * 1e-3 * fs))

    n_frames = 1 + (len(audio) - frame_len) // hop_len
    frames = np.stack([audio[i * hop_len : i * hop_len + frame_len] for i in range(n_frames)], axis=0)

    #Compute STFT (magnitude)
    freq, time, Zxx = scipy.signal.stft(frames, fs, nperseg=frame_len, noverlap=frame_len - hop_len, boundary='zeros')
    Zxx = Zxx[:, :, 0] #get the first channel
    graphs = build_time_slice_graphs(Zxx, r=4, eps=1e-12)
    print(f"Built {len(graphs)} time-slice graphs, each with {len(graphs[0].nodes)} nodes and {len(graphs[0].edges)} edges.")
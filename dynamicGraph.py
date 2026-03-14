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

def build_time_slice_graphs(Zxx, r=4):
    F, T = Zxx.shape
    graphs = []

    for k in range(T):
        #amplitudes for time slice k
        a = np.abs(Zxx[:, k]).astype(float)
        #a = Zxx[:, k].astype(float)
        #normalize
        den = float(np.sum(a*a) + eps)

        G = nx.Graph()  #undirected local-frequency graph
        G.graph["ref_phase"] = np.angle(Zxx[0, k])

        #nodes: frequency bins
        for i in range(F):
            #node id is the frequency-bin index i
            #store features as attributes
            log_power = 10.0 * np.log10(a[i]*a[i] + eps)
            G.add_node(i, amp=a[i], log_power=log_power)
           # G.add_node(i, coeff=Zxx[i, k])

        #edges: local neighborhood within +/- r
        #weight based on amplitude similarity (normalized dot product)
        for i in range(F):
            for d in range(1, r+1):
                j = i + d
                if j < F:
                    w = (a[i] * a[j]) / den
                    G.add_edge(i, j, weight=w)

        graphs.append(G)

    return graphs

def reconstruct_Zxx_from_graphs(graphs):
    T = len(graphs)
    F = graphs[0].number_of_nodes()
    Zxx_hat = np.zeros((F, T), dtype=np.complex128)

    for k, G in enumerate(graphs):
        amps = np.array([G.nodes[i]["amp"] for i in range(F)], dtype=float)
        phases = np.full(F, np.nan, dtype=float)

        # reference phase for node 0
        phases[0] = G.graph.get("ref_phase", 0.0)

        # propagate phases through the graph
        stack = [0]
        visited = {0}

        while stack:
            i = stack.pop()
            for j in G.neighbors(i):
                if j not in visited:
                    w = G.edges[i, j]["weight"]  # complex
                    delta = np.angle(w)          # phi_i - phi_j
                    phases[j] = phases[i] - delta
                    visited.add(j)
                    stack.append(j)

        # build STFT slice
        Zxx_hat[:, k]=amps*np.exp(1j*phases)
        #for i in range(F):
        #    Zxx_hat[i, k] = G.nodes[i]["coeff"]

    return Zxx_hat

def compute_snr(x, x_rec):
    L = min(len(x), len(x_rec))
    x = x[:L]
    x_rec = x_rec[:L]

    noise = x - x_rec

    snr = 10 * np.log10(
        np.sum(x**2) / (np.sum(noise**2) + 1e-12)
    )

    return snr

def compute_mse(x, x_rec):
    L = min(len(x), len(x_rec))
    return np.mean((x[:L] - x_rec[:L])**2)

def compare_spectorgram(x, x_rec, fs):
    f1, t1, S1 = scipy.spectrogram(x, fs)
    f2, t2, S2 = scipy.spectrogram(x_rec, fs)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(10*np.log10(S1+1e-12), aspect="auto", origin="lower")

    plt.subplot(1,2,2)
    plt.title("Reconstructed")
    plt.imshow(10*np.log10(S2+1e-12), aspect="auto", origin="lower")

    plt.show()

audio_files = ["tone.wav"]
for file in audio_files:
    x, fs = load_audio(file)
    #plot_audio(x, fs)
    frame_ms = 25
    hop_ms = 10
    frame_len = int(round(frame_ms * 1e-3 * fs))
    hop_len = int(round(hop_ms * 1e-3 * fs))

    #n_frames = 1 + (len(x) - frame_len) // hop_len
    #frames = np.stack([x[i * hop_len : i * hop_len + frame_len] for i in range(n_frames)], axis=0)


    #Build the graphs
    #TO DO: Add hann window
    freq, time, Zxx = scipy.signal.stft(x, fs, nperseg=frame_len, noverlap=frame_len-hop_len, boundary='zeros')
    #Zxx = Zxx[:, :, 0] #get the first channel

    graphs = build_time_slice_graphs(Zxx, r=4)
    print(f"Built {len(graphs)} time-slice graphs, each with {len(graphs[0].nodes)} nodes and {len(graphs[0].edges)} edges.")

    #Reconstruct the signal from the graphs
    Zxx_hat=reconstruct_Zxx_from_graphs(graphs)
    _, x_rec = scipy.signal.istft(Zxx_hat, fs=fs, noverlap=frame_len-hop_len)
    plot_audio(x_rec, fs)
    sf.write("./reconstructed_audio_signals/tone_reconstructed.wav", x_rec, fs)

    #Comparison metrics
    snr=compute_snr(x, x_rec)
    print("SNR: ", snr)
    mse=compute_mse(x, x_rec)
    print("MSE: ", mse)
    error = x[:len(x_rec)] - x_rec
    sf.write("./error_audio_signals/tone_error.wav", error, fs)
    #print(np.min(np.abs(Zxx_hat)), np.max(np.abs(Zxx_hat)), np.mean(np.abs(Zxx_hat)))

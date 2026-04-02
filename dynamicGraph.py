import torch
import torchaudio
import torch.nn.functional as F
from torch_geometric.data import Data


def audio_embedding_from_logmel(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 48,
):
    #Embedding = [mean(log-mel), std(log-mel)] over time.
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  #mono

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel = mel_transform(waveform)              #[1, n_mels, T]
    mel = mel.squeeze(0).transpose(0, 1)       #[T, n_mels]
    log_mel = torch.log(mel + 1e-6)

    mean = log_mel.mean(dim=0)                 #[n_mels]
    std = log_mel.std(dim=0, unbiased=False)   #[n_mels]

    emb = torch.cat([mean, std], dim=0)        #[2 * n_mels]
    return emb

def build_embedding_matrix(audio_list):
    embeddings = []
    for waveform, sr in audio_list:
        emb = audio_embedding_from_logmel(waveform, sr)
        embeddings.append(emb)

    x = torch.stack(embeddings, dim=0) 
    x = F.normalize(x, p=2, dim=1)
    return x

def build_knn_graph(x: torch.Tensor, k: int = 10):
    #cosine similarity matrix because x is normalized
    sim = x @ x.T  # [N, N]

    N = x.size(0)
    edges = []
    weights = []

    for i in range(N):
        vals, idx = torch.topk(sim[i], k=min(k + 1, N)) 
        for val, j in zip(vals.tolist(), idx.tolist()):
            if i == j:
                continue
            edges.append([i, j])
            weights.append([val])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_attr

def build_inter_audio_graph(audio_list, k: int = 10):
    x = build_embedding_matrix(audio_list)
    edge_index, edge_attr = build_knn_graph(x, k=k)

    graph = Data(
        x=x,                 
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return graph

files = ["tone.wav"]

audio_list = []
for path in files:
    waveform, sr = torchaudio.load(path)
    audio_list.append((waveform, sr))

G = build_inter_audio_graph(audio_list, k=2)

print(G)
print("Node feature shape:", G.x.shape)         
print("Edge index shape:", G.edge_index.shape)  
print("Edge attr shape:", G.edge_attr.shape)    
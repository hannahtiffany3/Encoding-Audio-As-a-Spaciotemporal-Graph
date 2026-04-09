import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch_geometric.data import Data
from laion_clap import CLAP_Module

clap=CLAP_Module(enable_fusion=False)
clap.load_ckpt()

def audio_embedding(waveform, sr):
    if waveform.shape[0] > 1:
        waveform=waveform.mean(dim=0, keepdim=True)

    if sr != 48000:
        waveform=torchaudio.functional.resample(waveform, sr, 48000)
    
    audio=waveform.squeeze(0).float()
    audio=audio.unsqueeze(0)
    emb=clap.get_audio_embedding_from_data(x=audio, use_tensor=True)
    return emb.squeeze(0)


def build_embedding_matrix(audio_list):
    embeddings=[]
    for waveform, sr in audio_list:
        emb=audio_embedding(waveform, sr)
        embeddings.append(emb)

    x=torch.stack(embeddings, dim=0) 
    x=F.normalize(x, p=2, dim=1)
    return x

def build_knn_graph(x: torch.Tensor, k: int = 10):
    #cosine similarity matrix because x is normalized
    sim=x @ x.T  #[N, N]

    N=x.size(0)
    edges=[]
    weights=[]

    for i in range(N):
        vals, idx=torch.topk(sim[i], k=min(k + 1, N)) 
        for val, j in zip(vals.tolist(), idx.tolist()):
            if i==j:
                continue
            edges.append([i, j])
            weights.append([val])

    edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr=torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_attr

def build_inter_audio_graph(audio_list, k: int = 10):
    x=build_embedding_matrix(audio_list)
    edge_index, edge_attr=build_knn_graph(x, k=k)

    graph = Data(
        x=x,                 
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return graph

files = ["tone.wav"]

audio_list = []
for file in files:
    path = "audio_signals/" + file
    waveform, sr = torchaudio.load(path)
    audio_list.append((waveform, sr))

G=build_inter_audio_graph(audio_list, k=2)

print(G)
print("Node feature shape:", G.x.shape)         
print("Edge index shape:", G.edge_index.shape)  
print("Edge attr shape:", G.edge_attr.shape)    
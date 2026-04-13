import numpy as np
import os
import json
import random
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
    for i, (waveform, sr) in enumerate(audio_list):
        print("Embedding audio " + str(i)) 
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

    print("Building graph")
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

def neighbors(G, i):
    src = G.edge_index[0]
    dst = G.edge_index[1]

    mask = (src == i)
    return dst[mask]

def neighbors_with_scores(G, i):
    src = G.edge_index[0]
    dst = G.edge_index[1]
    w = G.edge_attr.squeeze()

    mask = (src == i)

    return list(zip(dst[mask].tolist(), w[mask].tolist()))

def top_neighbors(G, i, k=5):
    pairs = neighbors_with_scores(G, i)
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return pairs[:k]

folders=["tone", "music", "singing", "speech"]

audio_list = []
#position in  the list indicates the node # in the graph
#maps number to audio name and type
labels=[]
filenames=[]
filenames_mapping={}

for i, folder in enumerate(folders):
    filepath="audio_signals/" + folder
    files=os.listdir(filepath)
    
    for i, file in enumerate(files):
        print("Loading Audio File "+ file)
        path = filepath + "/" + file
        waveform, sr = torchaudio.load(path)
        audio_list.append((waveform, sr))
        labels.append(folder)
        name=folder + str(i)
        filenames.append(name)
        filenames_mapping[name]=file

random.shuffle(audio_list)
G=build_inter_audio_graph(audio_list, k=5)
G.soundtype=labels
G.filename=filenames

print(G)
print("Node feature shape:", G.x.shape)         
print("Edge index shape:", G.edge_index.shape)  
print("Edge attr shape:", G.edge_attr.shape)    

sim_values = G.edge_attr.squeeze()

print(sim_values.min())
print(sim_values.mean())
print(sim_values.max())

torch.save(G, "graphData/graph1/graph1.pt")
with open("graphData/graph1/graph1.json", "w") as f:
    json.dump(filenames_mapping, f, indent=2)
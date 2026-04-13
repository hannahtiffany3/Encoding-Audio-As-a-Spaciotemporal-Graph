import torch
import sys
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

#Graph, colored by audio type
def printGraph(G):
    nx_graph = to_networkx(G, to_undirected=True, node_attrs=["soundtype", "filename"], edge_attrs=["edge_attr"])

    color_map={
        "tone": "blue",
        "music": "green",
        "singing": "yellow",
        "speech": "red"
    }
    node_colors = [color_map[nx_graph.nodes[n]['soundtype']] for n in nx_graph.nodes]

    plt.figure(figsize=(6,6))
    nx.draw(nx_graph, node_color=node_colors, node_size=100)
    plt.show()

#Heatmap
def heatmap(G):
    x=G.x
    x=torch.nn.functional.normalize(G.x, p=2, dim=1)
    sim=x @ x.T

    labels=G.soundtype
    names=G.filename

    order=sorted(range(len(labels)), key=lambda i: labels[i])

    sim_sorted=sim[order][:, order]
    names_sorted=[names[i] for i in order]

    plt.figure(figsize=(10, 8))
    plt.imshow(sim_sorted.cpu().detach().numpy(), aspect='auto')
    plt.colorbar(label="Cosine similarity")
    plt.title("Audio similarity heatmap (sorted)")
    plt.xticks(range(len(names_sorted)), names_sorted, rotation=90)
    plt.yticks(range(len(names_sorted)), names_sorted)
    plt.tight_layout()
    plt.show()

commands=sys.argv
G=torch.load("graphData/graph1/graph1.pt", weights_only=False)
if "p" in commands:
    printGraph(G)
if "h" in commands:
    heatmap(G)
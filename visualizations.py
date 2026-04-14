import torch
import sys
import json
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

#Graph, colored by audio type
def printGraph(G):
    nx_graph = to_networkx(G, to_undirected=True, node_attrs=["soundtype", "filename"], edge_attrs=["edge_attr"])
    mapping = {
        n: nx_graph.nodes[n]['filename']
        for n in nx_graph.nodes
    }

    color_map={
        "country": "blue",
        "hiphop": "green",
        "rnb": "yellow",
        "pop": "red",
        "rock": "grey"
    }
    node_colors = [color_map[nx_graph.nodes[n]['soundtype']] for n in nx_graph.nodes]

    plt.figure(figsize=(6,6))
    nx.draw(nx_graph, node_color=node_colors, labels=mapping, with_labels=True, node_size=500)
    plt.show()

#Heatmap
def heatmap(G):
    x=G.x
    x=torch.nn.functional.normalize(G.x, p=2, dim=1)
    sim=x @ x.T

    names=G.filename

    plt.figure(figsize=(10, 8))
    plt.imshow(sim.cpu().detach().numpy(), aspect='auto')
    plt.colorbar(label="Cosine similarity")
    plt.title("Audio similarity heatmap (unsorted)")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.tight_layout()
    plt.show()

    sim_np=sim.cpu().detach().numpy()
    #Convert similarity to distance
    dist=1.0-sim_np
    #Condensed distance matrix for scipy
    dist_condensed=squareform(dist, checks=False)
    #Hierarchical clustering
    Z=linkage(dist_condensed, method="average")
    #Get leaf order
    order=leaves_list(Z)
    #Reorder matrix and names
    sim_sorted=sim_np[order][:, order]
    names_sorted=[G.filename[i] for i in order]

    plt.figure(figsize=(10, 8))
    plt.imshow(sim_sorted, aspect="auto")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(len(names_sorted)), names_sorted, rotation=90)
    plt.yticks(range(len(names_sorted)), names_sorted)
    plt.title("Audio Heatmap Similarity (sorted)")
    plt.tight_layout()
    plt.show()


commands=sys.argv
G=torch.load("graphData/graph4/graph4.pt", weights_only=False)

if "p" in commands:
    printGraph(G)
if "h" in commands:
    heatmap(G)
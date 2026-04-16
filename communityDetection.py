import torch
import random
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import sys

def modularity_gain(graph, communities, node, target_community):
    internal_edges = sum(1 for neighbor in graph.neighbors(node) if communities[neighbor] == target_community)
    return internal_edges

def communityDetection(nx_graph):
    communities={node: node for node in nx_graph.nodes()}

    for _ in range(10):  #Arbitrary number of iterations
        nodes=list(nx_graph.nodes())
        random.shuffle(nodes)
        for node in nodes:
            best_community=communities[node]
            max_gain=0
            for neighbor in nx_graph.neighbors(node):
                target_community=communities[neighbor]
                gain=modularity_gain(nx_graph, communities, node, target_community)
                if gain>max_gain:
                    best_community=target_community
                    max_gain=gain
            communities[node]=best_community
    return communities

def visualize(nx_graph, mapping, communities):
    #Assign colors based on the community each node belongs to
    unique_communities=list(set(communities.values()))
    color_map = {community: i for i, community in enumerate(unique_communities)}
    node_colors = [color_map[communities[node]] for node in nx_graph.nodes()]

    #Draw the graph with communities highlighted
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(nx_graph, seed=42)
    nx.draw(
        nx_graph, pos, labels=mapping, with_labels=True, node_color=node_colors, 
        node_size=500, cmap=plt.cm.get_cmap('viridis', len(unique_communities))
    )

    plt.title("Graph with Community Detection", fontsize=14)
    plt.show()

def metrics(nx_graph, G, communities):
    modularity=nx.community.modularity(nx_graph, communities, weight="edge_attr")
    print("Modularity: ", modularity)
    x = G.x.detach().cpu().numpy()
    labels = G.soundtype
    ss=silhouette_score(x, labels)
    print("Sihlouette Score: ", ss)
    db=davies_bouldin_score(x, labels)
    print("Davies-Bouldin: ", db)

commands=sys.argv
G=torch.load("graphData/music10/music10.pt", weights_only=False)
G.weight = G.edge_attr.view(-1)
nx_graph=to_networkx(G, to_undirected=True, node_attrs=["soundtype", "filename"], edge_attrs=["weight"])
mapping = {
        n: nx_graph.nodes[n]['filename']
        for n in nx_graph.nodes
}
communities=communityDetection(nx_graph)
grouped={}
for k, v in communities.items():
    if v in grouped:
        grouped[v].append(k)
    else:
        grouped[v]=[]
        grouped[v].append(k)
groups=[]
for k, v in grouped.items():
    groups.append(v)


if "m" in commands:
    metrics(nx_graph, G, groups)
if "v" in commands:
    visualize(nx_graph, mapping, communities)
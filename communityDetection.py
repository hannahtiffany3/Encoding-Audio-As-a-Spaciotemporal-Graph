import torch
import random
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

def modularity_gain(graph, communities, node, target_community):
    internal_edges = sum(1 for neighbor in graph.neighbors(node) if communities[neighbor] == target_community)
    return internal_edges

def communityDetection(G):
    nx_graph=to_networkx(G, to_undirected=True, node_attrs=["soundtype", "filename"], edge_attrs=["edge_attr"])
    mapping = {
        n: nx_graph.nodes[n]['filename']
        for n in nx_graph.nodes
    }
    
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

G=torch.load("graphData/graph4/graph4.pt", weights_only=False)
communityDetection(G)

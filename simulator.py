import networkx as nx
import matplotlib.pyplot as plt
import random

# Create a financial networkcd
G = nx.DiGraph()

# Add nodes (participants)
nodes = ['NYSE', 'NASDAQ', 'Citadel', 'Retail_1', 'Retail_2', 'HedgeFund_1']
G.add_nodes_from(nodes)

# Add weighted edges (likelihood of flow)
edges = [
    ('Retail_1', 'Citadel', 0.8),
    ('Retail_2', 'Citadel', 0.6),
    ('Citadel', 'NYSE', 0.9),
    ('Citadel', 'NASDAQ', 0.7),
    ('HedgeFund_1', 'NYSE', 0.5),
    ('HedgeFund_1', 'NASDAQ', 0.5)
]
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# Simulate a single flow (e.g., order or info)
def simulate_flow(source, G):
    path = [source]
    current = source
    while True:
        neighbors = list(G.successors(current))
        if not neighbors:
            break
        probs = [G[current][n]['weight'] for n in neighbors]
        total = sum(probs)
        probs = [p / total for p in probs]
        current = random.choices(neighbors, weights=probs)[0]
        path.append(current)
    return path

# Example run
flow_path = simulate_flow('Retail_1', G)
print("Simulated Flow Path:", flow_path)

# Plot
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray')
nx.draw_networkx_edges(G, pos, edgelist=[(flow_path[i], flow_path[i+1]) for i in range(len(flow_path)-1)], width=3, edge_color='red')
plt.title("Simulated Market Flow")
plt.show()

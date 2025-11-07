import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces

def safe_shortest_path_length(graph, source, target, weight="travel_time"):
    try:
        return nx.shortest_path_length(graph, source, target, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")

def generate_synthetic_data(city="Chandigarh, India", n_deliveries=30, seed=42, save_path=None):
    print(f"📍 Generating synthetic data for {city} ({n_deliveries} deliveries)")
    ox.settings.use_cache = True
    graph = ox.graph_from_place(city, network_type="drive")
    nodes, _ = ox.graph_to_gdfs(graph)
    bounds = nodes.total_bounds
    np.random.seed(seed)
    lats = np.random.uniform(bounds[1], bounds[3], n_deliveries)
    lons = np.random.uniform(bounds[0], bounds[2], n_deliveries)
    delivery_nodes = ox.distance.nearest_nodes(graph, X=lons, Y=lats)
    deliveries = pd.DataFrame({
        "Delivery_ID": [f"D{i+1:03}" for i in range(n_deliveries)],
        "Latitude": nodes.loc[delivery_nodes, "y"].values,
        "Longitude": nodes.loc[delivery_nodes, "x"].values,
        "Package_Weight_kg": np.round(np.random.uniform(0.5, 10.0, n_deliveries), 2),
        "Priority": np.random.choice(["Low", "Medium", "High"], n_deliveries, p=[0.4, 0.4, 0.2])
    })
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        deliveries.to_csv(save_path, index=False)
        print(f"✅ Synthetic dataset saved to {save_path}")
    return graph, deliveries

def simulate_traffic(graph, save_path=None):
    print("🚦 Simulating dynamic traffic...")
    G = graph.copy()
    for u, v, k, data in G.edges(keys=True, data=True):
        base = data.get("length", 1)
        congestion = np.random.uniform(0.8, 2.0)
        data["travel_time"] = (base / 100) * congestion
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"✅ Graph reduced to largest connected component with {len(G)} nodes")
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(G, f)
        print(f"✅ Traffic graph saved as {save_path}")
    return G

class DeliveryEnv(gym.Env):
    def __init__(self, graph, deliveries, start_node):
        super().__init__()
        self.graph = graph
        self.start_node = start_node
        self.delivery_nodes = [
            ox.distance.nearest_nodes(graph, x, y)
            for y, x in zip(deliveries["Latitude"], deliveries["Longitude"])
        ]
        self.points_of_interest = [start_node] + self.delivery_nodes
        self.n_deliveries = len(self.delivery_nodes)
        self.n_points = len(self.points_of_interest)
        self.points = self.points_of_interest
        self.idx_to_node = {i: n for i, n in enumerate(self.points_of_interest)}
        self.dist_matrix = self._precompute_distances()
        self.observation_space = spaces.Box(low=0, high=1, shape=(1 + self.n_deliveries,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_deliveries)
        self.reset()

    def _precompute_distances(self):
        print("⏳ Pre-computing distance matrix...")
        matrix = np.full((self.n_points, self.n_points), float("inf"))
        for i in tqdm(range(self.n_points), desc="Distance Matrix"):
            for j in range(i, self.n_points):
                node_i = self.idx_to_node[i]
                node_j = self.idx_to_node[j]
                dist_ij = safe_shortest_path_length(self.graph, node_i, node_j)
                dist_ji = safe_shortest_path_length(self.graph, node_j, node_i)
                matrix[i, j] = dist_ij
                matrix[j, i] = dist_ji
        print("✅ Distance matrix complete.")
        return matrix

    def _get_state(self):
        state = np.zeros(1 + self.n_deliveries, dtype=np.float32)
        state[0] = self.current_idx / self.n_points
        state[1:] = self.visited_mask
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        self.visited_mask = np.zeros(self.n_deliveries, dtype=np.float32)
        self.remaining = self.n_deliveries
        self.total_time = 0.0
        self.route = [self.idx_to_node[0]]
        return self._get_state(), {}

    def step(self, action):
        if action < 0 or action >= self.n_deliveries:
            raise ValueError("Invalid action")
        if self.visited_mask[action] == 1:
            reward = -100.0
            done = self.remaining == 0
            return self._get_state(), reward, done, False, {}
        target_idx = action + 1
        travel_time = self.dist_matrix[self.current_idx, target_idx]
        if travel_time == float("inf"):
            reward = -200.0
            done = self.remaining == 0
            return self._get_state(), reward, done, False, {}
        self.total_time += travel_time
        self.current_idx = target_idx
        self.route.append(self.idx_to_node[self.current_idx])
        self.visited_mask[action] = 1.0
        self.remaining -= 1
        reward = -travel_time + 50.0
        done = self.remaining == 0
        if done:
            reward += 1000.0
        return self._get_state(), reward, done, False, {}

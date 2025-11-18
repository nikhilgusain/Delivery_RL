# 🚚 Dynamic Last-Mile Delivery Route Optimization via Deep Q-Learning and Classical Heuristics

**Author:** Aryan Dhasmana, Nikhil Gusian, Vishal Gupta
**Institution:** Chandigarh University  
**Year:** 2025  

---

## 🧭 Abstract

Efficient last-mile delivery routing remains a key challenge in urban logistics due to fluctuating traffic and time-sensitive demands.  
This repository implements and compares three algorithms — **Deep Q-Learning (DQN)**, **Traveling Salesman Problem (TSP)** solved via **2-Opt**, and the **Greedy Nearest Neighbor (NN)** — for **adaptive route planning** in a simulated city environment (Chandigarh, India).  

The model integrates **dynamic traffic simulation** using real-world OpenStreetMap (OSM) road data and evaluates GPU acceleration effects on DQN performance.  
Results show that while TSP remains superior under static conditions, DQN provides adaptive and scalable routing behavior under stochastic traffic, achieving up to **5× GPU speedup** in training and demonstrating the potential of reinforcement learning in **intelligent transportation systems (ITS)**.

---

## 🧩 Problem Formulation

The city road network is represented as a directed weighted graph:

$$
G = (V, E), \quad c_{ij} = \frac{L_{ij}}{100} \times \alpha_{ij}
$$

where $L_{ij}$ is edge length and $\alpha_{ij} \sim U(0.8, 2.0)$ models congestion.  
The goal is to minimize total travel time:

$$
\min_{\pi} T_{\text{total}} = \sum_{t=1}^{K} c_{p_{t-1}, p_t}
$$

subject to visiting each delivery node exactly once.

---

## 🧠 Algorithms Implemented

### A. Deep Q-Learning (DQN)
DQN learns an optimal policy $\pi^*(s) = \arg\max_a Q^*(s,a)$ through interaction with the environment.  
The Bellman optimality equation governs updates:


**Features:**
- Experience replay for sample decorrelation  
- Soft target network updates $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$  
- ε-greedy exploration with exponential decay  

---

### B. Traveling Salesman Problem (2-Opt Heuristic)
A local search heuristic that iteratively swaps two edges if doing so reduces total distance:

$$
d_{i,i+1}+d_{j,j+1} > d_{i,j}+d_{i+1,j+1}
$$

This continues until convergence to a local minimum.  
Complexity: $O(n^2)$ per iteration.

---

### C. Greedy Nearest Neighbor (NN)
At each step, the next delivery $v_{t+1}$ is chosen as:

$$
v_{t+1} = \arg\min_{v \in U_t} d(v_t, v)
$$

where $U_t$ is the set of unvisited nodes.  
This yields fast yet suboptimal routes, useful for lightweight or baseline routing scenarios.

---

## ⚙️ Project Structure

```
delivery_dqn_project/
│
├── envs/
│   └── delivery_env.py
├── training/
│   └── train_dqn.py
├── evaluation/
│   ├── evaluate_algorithms.py
│   └── runner.py
├── visualization/
│   └── charts.py
├── main.py
├── config.json
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

```bash
git clone https://github.com/<yourusername>/delivery_dqn_project.git
cd delivery_dqn_project
pip install -r requirements.txt
python main.py
```

Automatically detects GPU if available.

---

## 📊 Results Summary

| Algorithm | Total Travel Time | Evaluation Time | GPU Speedup |
|------------|------------------|-----------------|--------------|
| **DQN (Improved)** | 3085.21 | 3.12 | ×5 |
| **TSP (2-Opt)** | 1116.43 | 1.12 | — |
| **Greedy NN** | 1328.32 | 1.33 | — |

---

## 🗺️ Visualizations

- `algorithm_comparison_bars_extended.png` — Comparative performance bar chart  
- `dqn_training_comparison.png` — CPU vs GPU training times  
- `chandigarh_routes_comparison_extended.html` — Interactive route visualization  

---

## 📚 References

- Mnih, V. et al. (2015). *Human-Level Control Through Deep Reinforcement Learning.* Nature.  
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction.* MIT Press.  
- Boeing, G. (2017). *OSMnx: Modeling Urban Street Networks.* CEUS.  

---

## 🧠 Insights

- DQN adapts dynamically to congestion, outperforming deterministic methods under uncertainty.  
- GPU acceleration provides significant efficiency gains.  
- Classical heuristics remain useful for benchmarking and hybridization.

---

## 🧩 Future Work

- Multi-agent reinforcement learning for fleet coordination  
- Integration with Graph Neural Networks (GNNs)  
- Live traffic and CO₂-aware optimization  

---

## 📜 License

MIT License © 2025 Aryan Dhasmana

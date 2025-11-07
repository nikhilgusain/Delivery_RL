import time, torch, networkx as nx, numpy as np
from models.dqn import DQN

def evaluate_dqn(env, graph, model_path, label):
    print(f"Evaluating {label} from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state, _ = env.reset(); done = False
    with torch.no_grad():
        while not done:
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = model(s)
            q[0, s[0, 1:].bool()] = -float("inf")
            a = int(torch.argmax(q))
            state, _, t, tr, _ = env.step(a)
            done = t or tr
    route = env.route
    total_time = env.total_time
    print(f"✅ {label} total time: {total_time:.2f}")
    return {"route": route, "total_time": total_time}

def evaluate_greedy_nn(env, graph):
    print("Evaluating Greedy NN...")
    dist, visited, idx = env.dist_matrix, np.zeros(env.n_deliveries), 0
    route, total = [0], 0
    while not visited.all():
        options = [(dist[idx, i+1], i) for i in range(env.n_deliveries) if not visited[i]]
        d, j = min(options, key=lambda x: x[0])
        visited[j] = 1; total += d; idx = j+1; route.append(idx)
    total = float(total)
    print(f"✅ Greedy NN time: {total:.2f}")
    return {"route": [env.idx_to_node[i] for i in route], "total_time": total}

def evaluate_tsp(env, graph):
    print("Evaluating TSP (2-Opt)...")
    n, mat = env.n_points, env.dist_matrix
    route = list(range(n))
    improved = True
    def cost(r): return sum(mat[r[i], r[i+1]] for i in range(len(r)-1))
    best = cost(route)
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j - i == 1: continue
                new = route[:i] + route[i:j][::-1] + route[j:]
                new_c = cost(new)
                if new_c < best:
                    route, best, improved = new, new_c, True
    print(f"✅ TSP time: {best:.2f}")
    return {"route": [env.idx_to_node[i] for i in route], "total_time": best}

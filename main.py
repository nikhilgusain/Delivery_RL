import os, json, torch, time, pandas as pd
from envs.delivery_env import generate_synthetic_data, simulate_traffic, DeliveryEnv
from training.train_dqn import train_dqn
from evaluation.evaluate_algorithms import evaluate_dqn, evaluate_greedy_nn, evaluate_tsp
from visualization.charts import plot_training_comparison
from visualization.map_visualizer import generate_comparison_map
import osmnx as ox

def main():
    with open("config.json") as f: cfg = json.load(f)
    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["plots_dir"], exist_ok=True)
    os.makedirs(cfg["data_dir"], exist_ok=True)

    device_gpu = torch.device("cuda") if torch.cuda.is_available() else None
    device_cpu = torch.device("cpu")
    has_gpu = device_gpu is not None

    graph, deliveries = generate_synthetic_data(
        cfg["city"], cfg["n_deliveries"], cfg["seed"], os.path.join(cfg["data_dir"], "deliveries.csv"))
    graph = simulate_traffic(graph, os.path.join(cfg["data_dir"], "traffic.pkl"))
    warehouse = ox.distance.nearest_nodes(graph, X=76.7794, Y=30.7333)
    env = DeliveryEnv(graph, deliveries, warehouse)

    times = {}

    # --- DQN OLD (CPU + GPU)
    print("\n--- DQN (OLD) ---")
    t_cpu_old = train_dqn(env, device_cpu, "DQN (Old)", cfg, "models/dqn_old_cpu.pt",
                          n_episodes=500, lr=1e-4, eps_decay=1000)
    times["CPU (Old)"] = t_cpu_old
    if has_gpu:
        t_gpu_old = train_dqn(env, device_gpu, "DQN (Old)", cfg, "models/dqn_old_gpu.pt",
                              n_episodes=500, lr=1e-4, eps_decay=1000)
        times["GPU (Old)"] = t_gpu_old

    # --- DQN IMPROVED (CPU + GPU)
    print("\n--- DQN (IMPROVED) ---")
    t_cpu_imp = train_dqn(env, device_cpu, "DQN (Improved)", cfg, "models/dqn_improved_cpu.pt",
                          n_episodes=2000, lr=5e-5, eps_decay=2000)
    times["CPU (Improved)"] = t_cpu_imp
    if has_gpu:
        t_gpu_imp = train_dqn(env, device_gpu, "DQN (Improved)", cfg, "models/dqn_improved_gpu.pt",
                              n_episodes=2000, lr=5e-5, eps_decay=2000)
        times["GPU (Improved)"] = t_gpu_imp

    plot_training_comparison(times, cfg)

    # --- Evaluation ---
    print("\n⚖️ Evaluating Algorithms and Execution Times...\n")

    start_time = time.perf_counter()
    res_old = evaluate_dqn(env, graph, "models/dqn_old_cpu.pt", "DQN (Old)")
    exec_time_old = time.perf_counter() - start_time
    print(f"DQN (Old) evaluation took: {exec_time_old:.4f}s")

    start_time = time.perf_counter()
    res_imp = evaluate_dqn(env, graph, "models/dqn_improved_cpu.pt", "DQN (Improved)")
    exec_time_imp = time.perf_counter() - start_time
    print(f"DQN (Improved) evaluation took: {exec_time_imp:.4f}s")

    start_time = time.perf_counter()
    res_tsp = evaluate_tsp(env, graph)
    exec_time_tsp = time.perf_counter() - start_time
    print(f"TSP (2-Opt) evaluation took: {exec_time_tsp:.4f}s")
    
    start_time = time.perf_counter()
    res_greedy = evaluate_greedy_nn(env, graph)
    exec_time_greedy = time.perf_counter() - start_time
    print(f"Greedy (NN) evaluation took: {exec_time_greedy:.4f}s")

    print("\n--- Summary ---")
    summary = pd.DataFrame([
        {"Method": "DQN (Old)", "Total Travel Time": res_old["total_time"], "Execution Time (s)": exec_time_old},
        {"Method": "DQN (Improved)", "Total Travel Time": res_imp["total_time"], "Execution Time (s)": exec_time_imp},
        {"Method": "TSP (2-Opt)", "Total Travel Time": res_tsp["total_time"], "Execution Time (s)": exec_time_tsp},
        {"Method": "Greedy (NN)", "Total Travel Time": res_greedy["total_time"], "Execution Time (s)": exec_time_greedy}
    ])
    summary.to_csv("algorithm_metrics.csv", index=False)
    print(summary)

    generate_comparison_map(graph, deliveries, warehouse, {
        "DQN (Old)": res_old, "DQN (Improved)": res_imp, "TSP": res_tsp, "Greedy": res_greedy
    })

if __name__ == "__main__":
    main()
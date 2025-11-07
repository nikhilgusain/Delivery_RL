import folium
import os
import osmnx as ox
from branca.element import Template, MacroElement
import pandas as pd  # Added import for type hinting, though not strictly required

def generate_comparison_map(
    graph: 'networkx.MultiDiGraph', 
    deliveries: pd.DataFrame, 
    warehouse_node: int, 
    results: dict, 
    output_dir: str = "visualization/maps", 
    filename: str = "comparison_map.html"
):
    """
    Generates and saves a Folium map comparing different routing solutions.

    Args:
        graph (networkx.MultiDiGraph): The graph from OSMnx.
        deliveries (pd.DataFrame): DataFrame with "Latitude" and "Longitude" columns.
        warehouse_node (int): The OSMnx node ID for the warehouse.
        results (dict): A dictionary where keys are algorithm names (e.g., "TSP (2-Opt)")
                        and values are dicts containing a "route" (list of node IDs).
        output_dir (str): The directory to save the map in.
        filename (str): The name for the output HTML file.
    """
    
    nodes, _ = ox.graph_to_gdfs(graph)

    # --- 1. Dynamic Map Location ---
    try:
        warehouse_coords = [nodes.loc[warehouse_node].y, nodes.loc[warehouse_node].x]
    except KeyError:
        print(f"Warning: Warehouse node {warehouse_node} not found. Using default map center.")
        warehouse_coords = [30.7333, 76.7794] # Fallback location

    m = folium.Map(location=warehouse_coords, zoom_start=12, tiles="cartodb positron")

    # --- Color mapping ---
    # ** CHANGED: Cleaned up color keys to prevent mismatches **
    # Ensure the keys here (e.g., "DQN") EXACTLY match the keys in your `results` dict.
    colors = {
        "TSP (2-Opt)": "#1f77b4",  # Muted Blue
        "Greedy (NN)": "#ff7f0e",  # Safety Orange
        "DQN": "#d62728",         # Brick Red (Removed trailing space)
        "DQN (Altered)": "#9467bd" # Muted Purple
    }
    # These are from a common colorblind-friendly palette.

    # --- Draw routes ---
    print("Generating routes...")
    
    # Create a feature group for each algorithm to allow toggling
    for name, res in results.items():
        if "route" not in res:
            print(f"Skipping '{name}': 'route' key not found.")
            continue
            
        route_nodes = res["route"]
        coords = []
        for n in route_nodes:
            if n in nodes.index:
                coords.append((nodes.loc[n].y, nodes.loc[n].x))
            else:
                print(f"Warning: Node {n} from route '{name}' not in graph. Skipping node.")
                
        if len(coords) > 1:
            # Use .get() for a safe fallback to "gray" if a name is still not in the dict
            route_color = colors.get(name, "gray") 
            if route_color == "gray":
                 print(f"Warning: Route '{name}' not found in color map. Defaulting to gray.")

            folium.PolyLine(
                coords,
                color=route_color,
                weight=4,
                opacity=0.9,
                tooltip=name
            ).add_to(m)

    # --- Add delivery markers ---
    for _, r in deliveries.iterrows():
        folium.CircleMarker(
            [r["Latitude"], r["Longitude"]],
            radius=3,
            color="orange", # Kept this orange for contrast against routes
            fill=True,
            fill_opacity=0.9,
            tooltip="Delivery"
        ).add_to(m)

    # --- Warehouse marker ---
    folium.Marker(
        warehouse_coords,
        popup="Warehouse",
        icon=folium.Icon(color="black", icon="industry", prefix='fa')
    ).add_to(m)

    # --- 2. Dynamic Legend ---
    # The legend will automatically update
    legend_html_start = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 30px; right: 30px; 
        width: 180px; 
        background-color: white;
        border:2px solid grey; 
        border-radius:8px;
        z-index:9999;
        font-size:14px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        padding: 10px;">
        <b>🗺️ Legend</b><br>
    """
    
    legend_items = ""
    # Add route colors dynamically
    for name, color in colors.items():
        # ** CHANGED: Removed .strip() as keys are cleaner now **
        legend_items += f'<i style="background: {color}; width: 12px; height: 12px; float: left; margin-right: 6px; opacity: 0.9;"></i> {name}<br>'

    # Add fixed markers
    legend_items += '<i style="background: orange; width: 12px; height: 12px; float: left; margin-right: 6px; opacity: 0.9;"></i> Deliveries<br>'
    legend_items += '<i style="background: black; width: 12px; height: 12px; float: left; margin-right: 6px; opacity: 0.9;"></i> Warehouse'
    
    legend_html_end = """
    </div>
    {% endmacro %}
    """
    
    legend = MacroElement()
    legend._template = Template(legend_html_start + legend_items + legend_html_end)
    m.get_root().add_child(legend)

    # --- 3. Flexible Save Path ---
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    m.save(path)
    print(f"🗺️ Route comparison map with legend saved → {path}")
    
    return path
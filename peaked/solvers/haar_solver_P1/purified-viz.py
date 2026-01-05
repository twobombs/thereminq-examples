import re
import networkx as nx
import numpy as np
from vedo import *

# --- CONFIGURATION ---
FILENAME = 'layer_2_dimple_core.qasm'
BG_COLOR = "#0b0c10"        # Deep dark background
NODE_CMAP = "jet"           # Rainbow spectrum

# --- TRANSPARENCY SETTINGS ---
# Lower values = More transparent
NODE_ALPHA = 0.3            # Transparency of Qubit spheres
EDGE_ALPHA_BASE = 0.05      # Minimum opacity for weakest links
EDGE_ALPHA_SCALE = 0.5      # Scaling factor for stronger links

def get_cz_connections(filename):
    """Parses QASM file to extract CZ gate connections (edges)."""
    edges = []
    weights = {}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return [], {}

    pattern = re.compile(r"cz q\[(\d+)\],q\[(\d+)\]")
    for line in lines:
        match = pattern.search(line)
        if match:
            u, v = int(match.group(1)), int(match.group(2))
            pair = tuple(sorted((u, v)))
            edges.append(pair)
            weights[pair] = weights.get(pair, 0) + 1
    return edges, weights

def create_gradient_tube(p1, p2, c1, c2, radius, opacity):
    """
    Creates a Tube between p1 and p2 with a smooth color gradient.
    """
    t = Tube([p1, p2], r=radius, res=8)
    
    try:
        pts = t.points
        if callable(pts): 
            pts = pts()
    except TypeError:
        pts = t.points()

    vec = p2 - p1
    length_sq = np.dot(vec, vec)
    
    if length_sq < 1e-9: 
        return t.c(c1).alpha(opacity)
        
    scalars = np.dot(pts - p1, vec) / length_sq
    scalars = np.clip(scalars, 0.0, 1.0)
    
    t.pointdata["grad"] = scalars
    t.cmap([c1, c2], "grad")
    
    # Apply the requested opacity
    t.alpha(opacity)
    return t

def visualize_quantum_core_3d():
    print(f"Loading topology from {FILENAME}...")
    edges, weights = get_cz_connections(FILENAME)
    
    if not edges:
        print("No connections found. Check your QASM file.")
        return

    G = nx.Graph()
    G.add_edges_from(edges)
    
    print("Calculating 3D 'Spring' Layout...")
    pos = nx.spring_layout(G, dim=3, seed=42, k=0.6, iterations=200)

    max_idx = max(G.nodes())
    min_idx = min(G.nodes())

    def get_node_color_rgb(node_idx):
        val = (node_idx - min_idx) / (max_idx - min_idx + 1e-9)
        return color_map(val, name=NODE_CMAP, vmin=0, vmax=1)

    actors = []
    
    # 1. DRAW NODES (QUBITS)
    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    
    for node, coords in pos.items():
        deg = degrees[node]
        radius = 0.05 + (deg / max_deg) * 0.15
        
        sphere = Sphere(pos=coords, r=radius, res=24)
        
        c = get_node_color_rgb(node)
        # Use the new NODE_ALPHA setting for high transparency
        sphere.c(c).alpha(NODE_ALPHA).lighting('glossy')
        actors.append(sphere)
        
        label_pos = coords + np.array([0, radius + 0.05, 0])
        label = Text3D(str(node), pos=label_pos, s=0.06, c='white', depth=0)
        try:
            label.billboard()
        except AttributeError:
            pass
        actors.append(label)

    # 2. DRAW EDGES WITH GRADIENTS
    print("Generating Gradient Connections...")
    max_weight = max(weights.values()) if weights else 1
    
    for edge in G.edges():
        u, v = edge
        p1 = pos[u]
        p2 = pos[v]
        
        c1 = get_node_color_rgb(u)
        c2 = get_node_color_rgb(v)
        
        weight = weights.get(tuple(sorted(edge)), 1)
        
        thickness = 0.005 + (weight / max_weight) * 0.02
        
        # Calculate opacity using new transparency settings
        # Weakest links will be very faint (0.05), strongest will be semi-transparent (0.55)
        opacity = EDGE_ALPHA_BASE + (weight / max_weight) * EDGE_ALPHA_SCALE
        
        tube = create_gradient_tube(p1, p2, c1, c2, thickness, opacity)
        actors.append(tube)

    print("Launching Holographic View...")
    
    msg = Text2D(
        f"Quantum Core: {FILENAME}\n(High Transparency Mode)",
        pos="top-left", s=0.8, c="white", font="Calco"
    )
    
    plt = Plotter(bg=BG_COLOR, axes=0)
    plt.show(actors, msg, viewup='y', zoom=1.2, interactive=True)

if __name__ == "__main__":
    visualize_quantum_core_3d()

import torch
import networkx as nx
from vedo import Plotter, Sphere, Line, Text3D, Arrow
import numpy as np
import time
import math
from collections import defaultdict
import bisect

def get_spectrum_color(rank_percentile):
    """
    Maps a rank (0.0 to 1.0) to a Rainbow Spectrum.
    """
    v = max(0.0, min(1.0, rank_percentile))
    stops = [
        (0.0,  np.array([0.5, 0.0, 1.0])), # Purple
        (0.15, np.array([0.0, 0.0, 1.0])), # Blue
        (0.3,  np.array([0.0, 1.0, 1.0])), # Cyan
        (0.5,  np.array([0.0, 1.0, 0.0])), # Green
        (0.7,  np.array([1.0, 1.0, 0.0])), # Yellow
        (0.85, np.array([1.0, 0.5, 0.0])), # Orange
        (1.0,  np.array([1.0, 0.0, 0.0]))  # Red
    ]
    for i in range(len(stops) - 1):
        v1, c1 = stops[i]
        v2, c2 = stops[i+1]
        if v1 <= v <= v2:
            t = (v - v1) / (v2 - v1)
            return (1.0 - t) * c1 + t * c2
    return stops[-1][1]

def visualize_physics_rainbow(tensor_path='circuit_tensor.pt'):
    print(f"Loading {tensor_path}...")
    try:
        data = torch.load(tensor_path)
    except Exception as e:
        print(f"Error loading tensor: {e}")
        return

    num_gates = data.shape[0]
    
    # --- 1. Interaction Analysis ---
    print("Analyzing interactions...")
    interaction_counts = defaultdict(int)
    all_qubits = set()
    first_gate_for_qubit = {} 
    
    for i, row in enumerate(data):
        q1 = int(row[1].item())
        q2 = int(row[2].item())
        all_qubits.add(q1)
        
        if q1 not in first_gate_for_qubit: first_gate_for_qubit[q1] = i
        
        if q2 != -1:
            all_qubits.add(q2)
            if q2 not in first_gate_for_qubit: first_gate_for_qubit[q2] = i
            interaction_counts[q1] += 1
            interaction_counts[q2] += 1
            
    num_qubits = max(all_qubits) + 1
    
    # --- 2. Rank Logic ---
    all_counts = sorted([c for c in interaction_counts.values() if c > 0])
    cutoff_index = int(len(all_counts) * 0.5)
    cutoff_val = all_counts[cutoff_index] if all_counts else 0
    active_population = sorted([c for c in all_counts if c > cutoff_val])

    def get_rank_color_and_alpha(count):
        if count <= cutoff_val:
            return 'gray', 0.02, 1
        if not active_population: return 'gray', 0.02, 1
        
        rank_idx = bisect.bisect_left(active_population, count)
        percentile = rank_idx / len(active_population)
        col = get_spectrum_color(percentile)
        alpha = 0.1 + (0.2 * percentile)
        lw = 1 + (2 * percentile)
        return col, alpha, lw

    # --- 3. Topology ---
    dag = nx.DiGraph()
    last_gate_on_qubit = {} 
    for i in range(num_gates):
        dag.add_node(i)

    for i, row in enumerate(data):
        q1 = int(row[1].item())
        q2 = int(row[2].item())
        if q1 in last_gate_on_qubit: dag.add_edge(last_gate_on_qubit[q1], i, qubit=q1)
        last_gate_on_qubit[q1] = i
        if q2 != -1:
            if q2 in last_gate_on_qubit: dag.add_edge(last_gate_on_qubit[q2], i, qubit=q2)
            last_gate_on_qubit[q2] = i

    # --- 4. PHYSICS LAYOUT ---
    print("Running Physics Engine (Force-Directed Graph)...")
    try:
        layers = list(nx.topological_generations(dag))
    except:
        layers = [[i] for i in range(num_gates)]
        
    gate_layer_map = {}
    for lx, nodes in enumerate(layers):
        for n in nodes: gate_layer_map[n] = lx
    max_layer = len(layers) if layers else 1

    undirected_G = dag.to_undirected()
    yz_pos = nx.spring_layout(undirected_G, dim=2, k=0.6, iterations=100, seed=42)

    radius_scale = 1500.0 
    time_stretch = 0.03   

    actors = []
    final_pos = {}
    
    # --- 5. Render Gates & Wires ---
    gate_style = {
        0: 'gray', 1: 'tomato', 2: 'tomato', 3: 'tomato', 4: 'orange',
        5: 'yellow', 6: 'yellow', 7: 'gold', 8: 'gold',
        9: 'dodgerblue', 10: 'dodgerblue', 11: 'dodgerblue',
        12: 'cyan', 13: 'cyan', 14: 'cyan',
        15: 'mediumseagreen', 16: 'mediumseagreen', 17: 'mediumseagreen',
        22: 'purple', 23: 'magenta', 25: 'white', 26: 'darkgray',
        28: 'teal', 29: 'teal', 30: 'teal'
    }

    for i in range(num_gates):
        lx = gate_layer_map.get(i, 0)
        x = lx * time_stretch
        py, pz = yz_pos[i]
        y = py * radius_scale
        z = pz * radius_scale
        
        final_pos[i] = (x, y, z)
        
        gate_type = int(data[i][0].item())
        is_io = (lx == 0 or lx == max_layer - 1)
        c = gate_style.get(gate_type, 'gray')
        r = 12.0 if is_io else (5.0 if gate_type in [15,17,23] else 2.0)
        actors.append(Sphere(pos=final_pos[i], r=r, c=c))

    for start, end, attrs in dag.edges(data=True):
        p1 = final_pos[start]
        p2 = final_pos[end]
        q_id = attrs.get('qubit', 0)
        c, alpha, lw = get_rank_color_and_alpha(interaction_counts[q_id])
        actors.append(Line(p1, p2, c=c, alpha=alpha, lw=lw))

    # --- 6. Labels ---
    print("Attaching Labels...")
    for q_idx, first_gate_idx in first_gate_for_qubit.items():
        if first_gate_idx in final_pos:
            pos = final_pos[first_gate_idx]
            label_pos = (pos[0] - 20, pos[1], pos[2])
            
            count = interaction_counts[q_idx]
            col, _, _ = get_rank_color_and_alpha(count)
            
            # --- FIX: Type-safe check ---
            if isinstance(col, str) and col == 'gray': 
                col = 'white'
            
            actors.append(Text3D(f"q{q_idx}", pos=label_pos, s=30, c=col, justify='center'))

    # --- 7. Legend & Display ---
    leg_x, leg_y = 0, -radius_scale - 150
    actors.append(Text3D("PHYSICS LAYOUT + RANK HEATMAP", pos=(leg_x, leg_y + 60, 0), s=20.0, c='white'))
    
    bar_width = 200
    bar_height = 20
    steps = 50
    for i in range(steps):
        t = i / steps
        col = get_spectrum_color(t)
        x_pos = leg_x + (i * (bar_width / steps))
        actors.append(Line((x_pos, leg_y, 0), (x_pos, leg_y + bar_height, 0), c=col, lw=10))

    actors.append(Text3D("Lower 50%", pos=(leg_x, leg_y - 20, 0), s=15.0, c='gray'))
    actors.append(Text3D("Top 1%", pos=(leg_x + bar_width, leg_y - 20, 0), s=15.0, c='red'))
    
    actors.append(Arrow((0, radius_scale+50, 0), (50, radius_scale+50, 0), c='white', s=1.0))
    actors.append(Text3D("TIME ->", pos=(0, radius_scale+100, 0), s=20.0, c='white'))

    state = {'drag': False, 'mouse': (0,0), 'cam': None, 'foc': None}
    def on_key(evt):
        if evt.keypress == 's': vp.screenshot(f"qviz_physics_{int(time.time())}.png", scale=4)
    def on_click(evt):
        state['drag'] = True
        state['mouse'] = (evt.x, evt.y)
        state['cam'] = np.array(vp.camera.GetPosition())
        state['foc'] = np.array(vp.camera.GetFocalPoint())
        state['sc'] = np.linalg.norm(state['cam']-state['foc']) / (vp.window.get_size()[1]*0.5)
    def on_release(evt): state['drag'] = False
    def on_move(evt):
        if not state['drag']: return
        dx, dy = state['mouse'][0]-evt.x, state['mouse'][1]-evt.y
        cam = vp.camera
        vr = np.cross(np.array(cam.GetDirectionOfProjection()), np.array(cam.GetViewUp()))
        mov = (vr*dx*state['sc']) + (np.array(cam.GetViewUp())*dy*state['sc'])
        cam.SetPosition(state['cam']+mov)
        cam.SetFocalPoint(state['foc']+mov)
        vp.render()

    print("Displaying Physics Layout...")
    vp = Plotter(title="Quantum Physics Viz", axes=0, bg='black')
    vp.add_callback('KeyPress', on_key)
    vp.add_callback('RightButtonPress', on_click)
    vp.add_callback('RightButtonRelease', on_release)
    vp.add_callback('MouseMove', on_move)
    vp.show(actors, viewup='z', zoom=1.1)

if __name__ == "__main__":
    visualize_physics_rainbow()

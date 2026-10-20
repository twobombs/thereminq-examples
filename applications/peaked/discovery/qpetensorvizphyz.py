import torch
import networkx as nx
from vedo import Plotter, Sphere, Line, Text3D, Arrow
import numpy as np
import time

def visualize_extreme_circuit(tensor_path='circuit_tensor.pt'):
    print(f"Loading {tensor_path}...")
    try:
        data = torch.load(tensor_path)
    except Exception as e:
        print(f"Error loading tensor: {e}")
        return

    num_gates = data.shape[0]
    
    # --- 1. Topology ---
    dag = nx.DiGraph()
    last_gate_on_qubit = {} 
    for i in range(num_gates):
        dag.add_node(i)

    for i, row in enumerate(data):
        q1 = int(row[1].item())
        q2 = int(row[2].item())
        if q1 in last_gate_on_qubit:
            dag.add_edge(last_gate_on_qubit[q1], i)
        last_gate_on_qubit[q1] = i
        if q2 != -1:
            if q2 in last_gate_on_qubit:
                dag.add_edge(last_gate_on_qubit[q2], i)
            last_gate_on_qubit[q2] = i

    print("Calculating layers...")
    try:
        layers = list(nx.topological_generations(dag))
    except:
        layers = [[i] for i in range(num_gates)]
        
    gate_layer_map = {}
    for layer_idx, nodes_in_layer in enumerate(layers):
        for node in nodes_in_layer:
            gate_layer_map[node] = layer_idx
    max_layer = len(layers) if layers else 1

    # --- 2. Physics Layout ---
    print("Untangling cross-sections...")
    undirected_G = dag.to_undirected()
    # Slightly lower k to keep the cloud cohesive despite the massive scaling
    yz_pos = nx.spring_layout(undirected_G, dim=2, k=0.6, iterations=100, seed=42)

    # --- 3. Style Definitions ---
    gate_style = {
        0: ('gray', 'Identity'),
        1: ('tomato', 'Pauli X'), 2: ('tomato', 'Pauli Y'), 3: ('tomato', 'Pauli Z'),
        4: ('orange', 'Hadamard'),
        5: ('yellow', 'S Gate'), 6: ('yellow', 'S Dagger'),
        7: ('gold', 'T Gate'), 8: ('gold', 'T Dagger'),
        9: ('dodgerblue', 'RX'), 10: ('dodgerblue', 'RY'), 11: ('dodgerblue', 'RZ'),
        12: ('cyan', 'U1'), 13: ('cyan', 'U2'), 14: ('cyan', 'U3'),
        15: ('mediumseagreen', 'CX'), 16: ('mediumseagreen', 'CY'), 17: ('mediumseagreen', 'CZ'),
        22: ('purple', 'SWAP'),
        23: ('magenta', 'Toffoli (CCX)'),
        25: ('white', 'Measure'),
        26: ('darkgray', 'Barrier'),
        28: ('teal', 'General U'),
        29: ('teal', 'CP'), 30: ('teal', 'CU')
    }

    actors = []
    
    # --- 4. EXTREME SCALING ---
    # Width 10x larger, Length 100x smaller
    time_stretch = 0.03    # Compressed Time
    radius_scale = 150.0   # Expanded Width
    
    final_pos = {}
    
    # Legend tracking
    legend_entries = []
    found_gate_types = set()
    has_input_layer = False
    has_output_layer = False

    # --- 5. Draw Gates ---
    for i in range(num_gates):
        layer_x = gate_layer_map.get(i, 0)
        y, z = yz_pos[i]
        pos = (layer_x * time_stretch, y * radius_scale, z * radius_scale)
        final_pos[i] = pos
        
        gate_type = int(data[i][0].item())
        found_gate_types.add(gate_type)
        
        is_input = (layer_x == 0)
        is_output = (layer_x == max_layer - 1)
        
        if is_input: has_input_layer = True
        if is_output: has_output_layer = True
        
        # Color Logic
        style = gate_style.get(gate_type, ('gray', 'Unknown'))
        c = style[0]
        r = 0.2

        if is_input:
            c, r = 'gold', 0.5
        elif is_output:
            c, r = 'red', 0.5
        elif gate_type in [15, 17, 23]: 
            r = 0.3
        
        actors.append(Sphere(pos=pos, r=r, c=c))

    # --- 6. Draw Wires ---
    for start, end in dag.edges():
        p1 = final_pos[start]
        p2 = final_pos[end]
        progress = gate_layer_map.get(start, 0) / max_layer
        wire_color = (progress, 0.6, 1.0 - progress) 
        # Reduced opacity because wires will be very dense now
        actors.append(Line(p1, p2, c=wire_color, alpha=0.2))

    # --- 7. Build Legend ---
    print("Building Legend...")
    if has_input_layer: legend_entries.append(('gold', 'Input Layer'))
    for g_id in sorted(list(found_gate_types)):
        if g_id in gate_style:
            color, name = gate_style[g_id]
            legend_entries.append((color, name))
    if has_output_layer: legend_entries.append(('red', 'Output Layer'))

    # Place Legend safely below the massive width
    legend_x = 0
    legend_y = -radius_scale - 20 
    
    actors.append(Text3D("LEGEND", pos=(legend_x, legend_y + 5, 0), s=2.0, c='white'))

    for idx, (color, label) in enumerate(legend_entries):
        y_pos = legend_y - (idx * 5)
        actors.append(Sphere(pos=(legend_x, y_pos, 0), r=1.0, c=color))
        actors.append(Text3D(label, pos=(legend_x + 3, y_pos, 0), s=1.5, c='white'))

    # Visual Guides (Time Arrow)
    # Arrow length scaled to fit the compressed time, but visible
    arrow_len = max(5.0, max_layer * time_stretch) 
    actors.append(Arrow((0, radius_scale + 10, 0), (arrow_len, radius_scale + 10, 0), c='white', s=0.1))
    actors.append(Text3D("TIME ->", pos=(0, radius_scale + 15, 0), s=2.0, c='white'))

    # --- 8. Interaction & Display ---
    interaction_state = {'dragging': False, 'start_mouse': (0,0), 'start_cam': None, 'start_foc': None}

    def on_keypress(event):
        if event.keypress == 's':
            fn = f"qviz_extreme_{int(time.time())}.png"
            print(f"Saving {fn}...")
            vp.screenshot(fn, scale=4)

    def on_click(evt):
        interaction_state['dragging'] = True
        interaction_state['start_mouse'] = (evt.x, evt.y)
        interaction_state['start_cam'] = np.array(vp.camera.GetPosition())
        interaction_state['start_foc'] = np.array(vp.camera.GetFocalPoint())
        dist = np.linalg.norm(interaction_state['start_cam'] - interaction_state['start_foc'])
        interaction_state['scale'] = dist / (vp.window.get_size()[1] * 0.5)

    def on_release(evt): interaction_state['dragging'] = False

    def on_move(evt):
        if not interaction_state['dragging']: return
        dx, dy = interaction_state['start_mouse'][0]-evt.x, interaction_state['start_mouse'][1]-evt.y
        cam = vp.camera
        vr = np.cross(np.array(cam.GetDirectionOfProjection()), np.array(cam.GetViewUp()))
        mov = (vr*dx*interaction_state['scale']) + (np.array(cam.GetViewUp())*dy*interaction_state['scale'])
        cam.SetPosition(interaction_state['start_cam']+mov)
        cam.SetFocalPoint(interaction_state['start_foc']+mov)
        vp.render()

    print("Displaying Extreme Layout...")
    vp = Plotter(title="Quantum Viz (Extreme Scale)", axes=0, bg='black')
    vp.add_callback('KeyPress', on_keypress)
    vp.add_callback('RightButtonPress', on_click)
    vp.add_callback('RightButtonRelease', on_release)
    vp.add_callback('MouseMove', on_move)
    vp.show(actors, viewup='z', zoom=1.1)

if __name__ == "__main__":
    visualize_extreme_circuit()

import torch
import networkx as nx
from vedo import Plotter, Sphere, Line, Text3D, Arrow
import numpy as np
import time

def visualize_dark_circuit(tensor_path='circuit_tensor.pt'):
    print(f"Loading {tensor_path}...")
    try:
        data = torch.load(tensor_path)
    except Exception as e:
        print(f"Error loading tensor: {e}")
        return

    num_gates = data.shape[0]
    
    # --- 1. Topology & Layers ---
    dag = nx.DiGraph()
    last_gate_on_qubit = {} 
    for i in range(num_gates):
        dag.add_node(i)

    # Build dependency graph
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
    layers = list(nx.topological_generations(dag))
    gate_layer_map = {}
    for layer_idx, nodes_in_layer in enumerate(layers):
        for node in nodes_in_layer:
            gate_layer_map[node] = layer_idx
    max_layer = len(layers)

    # --- 2. Physics Layout ---
    print("Untangling cross-sections...")
    undirected_G = dag.to_undirected()
    yz_pos = nx.spring_layout(undirected_G, dim=2, k=0.8, iterations=200, seed=42)

    # --- 3. Render Setup ---
    print("Building Dark Mode Scene...")
    actors = []
    
    time_stretch = 3.0    
    radius_scale = 15.0   

    gate_colors = {
        0: 'tomato',        # X
        1: 'orange',        # H
        2: 'dodgerblue',    # U1
        3: 'mediumseagreen',# CX
        4: 'purple'         # SWAP
    }

    final_pos = {}

    # Gates
    for i in range(num_gates):
        layer_x = gate_layer_map[i]
        y, z = yz_pos[i]
        pos = (layer_x * time_stretch, y * radius_scale, z * radius_scale)
        final_pos[i] = pos
        
        gate_type = int(data[i][0].item())
        
        is_input = (layer_x == 0)
        is_output = (layer_x == max_layer - 1)
        
        if is_input:
            c = 'gold'
            r = 0.6  
        elif is_output:
            c = 'red'
            r = 0.6  
        else:
            c = gate_colors.get(gate_type, 'gray')
            r = 0.25 if gate_type == 3 else 0.18

        actors.append(Sphere(pos=pos, r=r, c=c))

    # Wires
    for start, end in dag.edges():
        p1 = final_pos[start]
        p2 = final_pos[end]
        progress = gate_layer_map[start] / max_layer
        wire_color = (progress, 0.6, 1.0 - progress) 
        actors.append(Line(p1, p2, c=wire_color, alpha=0.5, lw=1))

    # --- 4. Manual Legend ---
    legend_start_x = 0
    legend_y = -radius_scale - 15 
    spacing = 4
    text_color = 'white'

    legend_items = [
        ('gold', "Input Layer"),
        ('tomato', "X Gate"),
        ('orange', "Hadamard"),
        ('dodgerblue', "Phase (U1)"),
        ('mediumseagreen', "CNOT (CX)"),
        ('purple', "SWAP"),
        ('red', "Output Layer"),
    ]

    actors.append(Text3D("LEGEND", pos=(legend_start_x, legend_y + 3, 0), s=1.2, c=text_color))

    for idx, (color, label) in enumerate(legend_items):
        item_pos = (legend_start_x, legend_y - (idx * spacing), 0)
        actors.append(Sphere(pos=item_pos, r=0.8, c=color))
        label_pos = (legend_start_x + 2, legend_y - (idx * spacing) - 0.3, 0)
        actors.append(Text3D(label, pos=label_pos, s=0.8, c=text_color))

    # Time Arrow
    actors.append(Arrow((0, radius_scale + 5, 0), (20, radius_scale + 5, 0), c=text_color, s=0.08)) 
    actors.append(Text3D("TIME / DEPTH", pos=(0, radius_scale + 8, 0), s=1, c=text_color))

    # --- 5. Interaction State (Globals for tracking) ---
    interaction_state = {
        'dragging': False,
        'start_mouse': (0, 0),
        'start_cam_pos': np.array([0.,0.,0.]),
        'start_focal': np.array([0.,0.,0.])
    }

    # --- Callbacks ---
    def on_keypress(event):
        if event.keypress == 's':
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"qpe_dark_{timestamp}.png"
            print(f"Taking screenshot: {filename}...")
            vp.screenshot(filename, scale=4)
            print(f"Saved!")

    def on_right_click(event):
        # Start Dragging
        interaction_state['dragging'] = True
        interaction_state['start_mouse'] = (event.x, event.y)
        
        # Record initial camera state
        cam = vp.camera
        interaction_state['start_cam_pos'] = np.array(cam.GetPosition())
        interaction_state['start_focal'] = np.array(cam.GetFocalPoint())
        
        # Calculate scaling factor based on distance to object
        # This makes the pan feel 1:1 with the mouse cursor
        dist = np.linalg.norm(interaction_state['start_cam_pos'] - interaction_state['start_focal'])
        interaction_state['scale_factor'] = dist / (vp.window.get_size()[1] * 0.5)

    def on_right_release(event):
        interaction_state['dragging'] = False

    def on_mouse_move(event):
        if not interaction_state['dragging']:
            return

        # Calculate Mouse Delta
        cur_x, cur_y = event.x, event.y
        start_x, start_y = interaction_state['start_mouse']
        dx = start_x - cur_x  # Inverted for natural drag feel
        dy = start_y - cur_y

        # Get Camera Vectors
        cam = vp.camera
        view_normal = np.array(cam.GetDirectionOfProjection())
        view_up = np.array(cam.GetViewUp())
        
        # Calculate 'Right' vector (Cross Product of Normal and Up)
        view_right = np.cross(view_normal, view_up)
        
        # Calculate World Movement
        scale = interaction_state['scale_factor']
        move_vec = (view_right * dx * scale) + (view_up * dy * scale)
        
        # Apply new positions
        new_pos = interaction_state['start_cam_pos'] + move_vec
        new_focal = interaction_state['start_focal'] + move_vec
        
        cam.SetPosition(new_pos)
        cam.SetFocalPoint(new_focal)
        
        # Force redraw
        vp.render()

    # --- Display ---
    print("Displaying...")
    print(" - Right-Click Drag: PAN")
    print(" - 's' Key: SCREENSHOT")
    
    vp = Plotter(title="Quantum Phase Estimation (Dark Mode)", axes=0, bg='black')
    
    # Register Callbacks
    vp.add_callback('KeyPress', on_keypress)
    vp.add_callback('RightButtonPress', on_right_click)
    vp.add_callback('RightButtonRelease', on_right_release)
    vp.add_callback('MouseMove', on_mouse_move)
    
    vp.show(actors, viewup='z', zoom=1.1)

if __name__ == "__main__":
    visualize_dark_circuit()

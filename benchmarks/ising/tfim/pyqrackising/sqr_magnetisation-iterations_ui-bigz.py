# -*- coding: utf-8 -*-
"""
An interactive 3D visualizer for the Transverse Field Ising Model (TFIM)
simulation output from iterations_cli.py.

This version supports multiple, distinct full-grid scans (J-h and Z-Theta)
and dynamically reconfigures the 3D plot for each. It uses multiprocessing 
for responsive UI and fast, parallel computations.
"""
import subprocess
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from vedo import Plotter, Points, Mesh, Text2D, ScalarBar
import threading
import queue
import sys
import multiprocessing as mp

# --- Simulation Runners ---

def _execute_simulation(params):
    """
    Core function to execute the CLI script and parse the output.
    This is a top-level function so it can be 'pickled' by multiprocessing.
    Returns a tuple: (result_type, data)
    """
    command = [sys.executable, 'sqr_magnetisation-iterations_cli.py']
    for key, value in params.items():
        command.extend([f'--{key}', str(value)])

    try:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(
            command, capture_output=True, text=True, check=True, creationflags=creationflags
        )

        output_lines = result.stdout.strip().split('\n')
        samples_line = ""
        time_line = "Calculation took: N/A"

        try:
            samples_index = output_lines.index("## Output Samples (Decimal Comma-Separated) ##")
            samples_line = output_lines[samples_index + 1]
        except (ValueError, IndexError):
            pass

        for line in output_lines:
            if line.startswith("Calculation took:"):
                time_line = line
                break

        if not samples_line:
            return ('error', "Could not find sample output in simulation results.")

        avg_magnetization = float(samples_line.replace(',', '.'))
        return ('success', (params, (avg_magnetization, time_line)))

    except FileNotFoundError:
        return ('error', "Error: 'sqr_magnetisation-iterations_cli.py' not found.")
    except subprocess.CalledProcessError as e:
        return ('error', f"Simulation failed with error:\n{e.stderr}")
    except Exception as e:
        return ('error', f"An unexpected error occurred: {e}")

def run_simulation_thread(params, result_queue):
    """Thread target for running a single simulation."""
    result_type, data = _execute_simulation(params)
    result_queue.put((result_type, data))

def run_scan_worker(scan_type, fixed_params, result_queue):
    """
    Worker to run a full grid scan in parallel. Dispatches to the correct
    parameter space based on scan_type.
    """
    all_params = []
    if scan_type == 'J_vs_h':
        j_range = np.arange(-2.0, 2.0 + 0.1, 0.1)
        h_range = np.arange(-4.0, 4.0 + 0.1, 0.1)
        for h_val in h_range:
            for j_val in j_range:
                p = fixed_params.copy()
                p['J'] = round(j_val, 2)
                p['h'] = round(h_val, 2)
                all_params.append(p)
    elif scan_type == 'z_vs_theta':
        z_range = np.arange(-8, 8 + 1, 1) # Use integer steps for z
        theta_range = np.arange(-np.pi / 2, np.pi / 2 + 0.1, 0.1)
        for theta_val in theta_range:
            for z_val in z_range:
                p = fixed_params.copy()
                p['z'] = int(z_val) # Ensure z is an integer
                p['theta'] = round(theta_val, 4)
                all_params.append(p)

    if not all_params:
        result_queue.put(('error', 'Invalid scan type specified.'))
        return

    total_points = len(all_params)
    count = 0
    try:
        multiplier = 3
        num_processes = mp.cpu_count() * multiplier
        print(f"Initializing pool with {num_processes} workers for {total_points} simulations...")

        with mp.Pool(processes=num_processes) as pool:
            for result_type, data in pool.imap_unordered(_execute_simulation, all_params):
                count += 1
                status_msg = f"Scanning... {count}/{total_points}"
                result_queue.put(('status', status_msg))
                result_queue.put((result_type, data))
        
        result_queue.put(('status', f"Scan complete. Plotted {total_points} points."))
    except Exception as e:
        result_queue.put(('error', f"A multiprocessing error occurred: {e}"))

# --- Vedo Plotter Process ---
def vedo_plotter_process(data_queue):
    """
    This function runs in a separate process to handle all vedo plotting.
    It can now dynamically change its axes and plot different surfaces.
    """
    # Initialize with default axes (style 1) to ensure they are always present.
    plt = Plotter(bg='black', title="TFIM Phase Visualizer", axes=1)

    # Data storage
    plotted_points_data = []
    scan_points_data = {}  # For the grid scan: (x_val, y_val) -> z_val
    scan_keys = ('J', 'h') # Default scan parameters

    # Vedo actors
    current_actor = None # Can be Points or Mesh
    info_text_actor = Text2D("Initializing...", pos='top-left', s=1.1, bg='gray', alpha=0.7)
    scalar_bar_actor = None
    
    plt.show(info_text_actor, interactive=False)

    def update_plot(timer_id):
        nonlocal current_actor, info_text_actor, scalar_bar_actor, scan_keys
        try:
            while not data_queue.empty():
                command, data = data_queue.get_nowait()
                
                if command == 'setup_plot':
                    # Manually remove old plot actors instead of calling plt.clear().
                    # This preserves the main axes object so it doesn't disappear.
                    plt.remove(current_actor, scalar_bar_actor)
                    current_actor = None
                    scalar_bar_actor = None
                    plotted_points_data.clear()
                    scan_points_data.clear()

                    scan_keys = (data['x_key'], data['y_key'])
                    axes_opts = dict(
                        xtitle=data['xtitle'], ytitle=data['ytitle'], ztitle='Avg. Measured Value',
                        zlabel_size=0.02, ylabel_size=0.02, xlabel_size=0.02,
                        xrange=data['xrange'], yrange=data['yrange'], zrange=(0.0, 1.0)
                    )
                    # This tells vedo what options to use for the axes on the next render.
                    plt.axes_options = axes_opts
                    info_text_actor.text("Ready for new scan.")
                    plt.reset_camera()


                elif command == 'add':
                    params, (avg_z, time_str) = data
                    x_val = params[scan_keys[0]]
                    y_val = params[scan_keys[1]]

                    new_point = (x_val, y_val, avg_z)
                    plotted_points_data.append(new_point)
                    scan_points_data[(x_val, y_val)] = avg_z

                    if isinstance(current_actor, Mesh): continue # Don't plot points over a finished surface

                    plt.remove(current_actor)
                    points_array = np.array(plotted_points_data)
                    z_values = points_array[:, 2]
                    current_actor = Points(points_array, r=5).cmap('viridis', z_values, vmin=0.0, vmax=1.0)

                    plt.remove(scalar_bar_actor)
                    scalar_bar_actor = ScalarBar(current_actor, title="Avg. Measured Value", pos=((0.85, 0.4), (0.9, 0.9)))
                    info_text = (f"Plotted points: {len(plotted_points_data)}\n"
                                 f"Last: {scan_keys[0]}={x_val:.2f}, {scan_keys[1]}={y_val:.2f}\n"
                                 f"Avg. Value: {avg_z:.4f}\n{time_str}")
                    info_text_actor.text(info_text)

                    plt.add(current_actor, scalar_bar_actor)

                elif command == 'scan_complete':
                    if len(scan_points_data) < 4: continue

                    print("Scan complete. Generating surface mesh...")
                    x_coords = sorted(list(set(p[0] for p in scan_points_data.keys())))
                    y_coords = sorted(list(set(p[1] for p in scan_points_data.keys())))
                    
                    if len(x_coords) < 2 or len(y_coords) < 2: continue

                    vertices = [[x, y, scan_points_data.get((x, y), 0.0)] for y in y_coords for x in x_coords]
                    
                    faces = []
                    nx, ny = len(x_coords), len(y_coords)
                    for i in range(ny - 1):
                        for j in range(nx - 1):
                            p1, p2 = i * nx + j, i * nx + j + 1
                            p3, p4 = (i + 1) * nx + j + 1, (i + 1) * nx + j
                            faces.append([p1, p2, p3, p4])

                    plt.remove(current_actor)
                    mesh_actor = Mesh([vertices, faces]).lighting('glossy')
                    mesh_actor.cmap('viridis', np.array(vertices)[:, 2], vmin=0.0, vmax=1.0)
                    current_actor = mesh_actor
                    
                    plt.remove(scalar_bar_actor)
                    scalar_bar_actor = ScalarBar(current_actor, title="Avg. Measured Value", pos=((0.85, 0.4), (0.9, 0.9)))
                    info_text_actor.text(f"Surface plot with {len(vertices)} vertices.\nScan complete.")
                    
                    plt.add(current_actor, scalar_bar_actor)
                    print("Surface mesh rendered.")

                elif command == 'clear':
                    plotted_points_data.clear()
                    scan_points_data.clear()
                    plt.remove(current_actor, scalar_bar_actor)
                    current_actor, scalar_bar_actor = None, None
                    info_text_actor.text("Plot cleared.")
                    # We add the info_text_actor back to ensure it's visible
                    plt.add(info_text_actor)
                    plt.reset_camera()

                elif command == 'close':
                    plt.close()
                    return

        except Exception as e:
            print(f"Error in vedo process: {e}", file=sys.stderr)
        
        plt.render()

    plt.add_callback('timer', update_plot)
    plt.timer_callback('create', dt=100)
    plt.interactive().close()

# --- GUI Class ---
class ControlPanel:
    def __init__(self, root, param_specs, data_queue):
        self.root = root
        self.param_specs = param_specs
        self.data_queue = data_queue
        self.sim_result_queue = queue.Queue()
        self.param_vars = {}
        self.scan_buttons = {}

        self.root.title("TFIM Control Panel")
        self.root.geometry("400x650")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready. Move a slider or start a scan.")
        self._create_controls(main_frame)
        
        self.root.after(100, self.check_sim_queue)

    def _create_controls(self, parent):
        slider_frame = ttk.LabelFrame(parent, text="Simulation Parameters", padding=10)
        slider_frame.pack(fill=tk.X, expand=True)
        
        for name, spec in self.param_specs.items():
            frame = ttk.Frame(slider_frame)
            frame.pack(fill=tk.X, pady=5)
            label = ttk.Label(frame, text=f"{name.replace('_', ' ').title()}:", width=12, anchor=tk.W)
            label.pack(side=tk.LEFT)

            var = tk.DoubleVar(value=spec['default'])
            self.param_vars[name] = var

            value_label = ttk.Label(frame, text=f"{var.get():.2f}", width=6)
            value_label.pack(side=tk.RIGHT, padx=(5, 0))

            command = lambda val, l=value_label, v=var, s=spec: \
                l.config(text=f"{v.get():.2f}" if s['resolution'] < 1 else f"{int(v.get())}")

            slider = ttk.Scale(frame, from_=spec['from_'], to=spec['to'], orient=tk.HORIZONTAL, variable=var, command=command)
            slider.pack(fill=tk.X, expand=True)
            slider.bind("<ButtonRelease-1>", self.run_simulation_from_sliders)

        button_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        button_frame.pack(fill=tk.X, pady=20)

        self.scan_buttons['J_vs_h'] = ttk.Button(button_frame, text="Run Scan (J vs h)", command=lambda: self.start_full_scan('J_vs_h'))
        self.scan_buttons['J_vs_h'].pack(side=tk.LEFT, expand=True, padx=5, ipady=5)

        self.scan_buttons['z_vs_theta'] = ttk.Button(button_frame, text="Run Scan (Z vs Theta)", command=lambda: self.start_full_scan('z_vs_theta'))
        self.scan_buttons['z_vs_theta'].pack(side=tk.LEFT, expand=True, padx=5, ipady=5)
        
        clear_button = ttk.Button(button_frame, text="Clear Plot", command=self.clear_plot)
        clear_button.pack(side=tk.RIGHT, expand=True, padx=5, ipady=5)

        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def run_simulation_from_sliders(self, event=None):
        # This function is for single point plotting, which assumes a J-h plot.
        # We should reset the plot to J-h mode if it was in another mode.
        self.data_queue.put(('setup_plot', {
            'xtitle': 'J (Coupling)', 'ytitle': 'h (Transverse Field)',
            'xrange': (self.param_specs['J']['from_'], self.param_specs['J']['to']),
            'yrange': (self.param_specs['h']['from_'], self.param_specs['h']['to']),
            'x_key': 'J', 'y_key': 'h'
        }))
        
        current_params = {name: var.get() for name, var in self.param_vars.items()}
        # Ensure parameters that must be integers are cast correctly
        for key in ['z', 't', 'n_qubits']:
            if key in current_params: current_params[key] = int(current_params[key])

        self.status_var.set("Running single simulation...")
        threading.Thread(target=run_simulation_thread, args=(current_params, self.sim_result_queue), daemon=True).start()
        
    def start_full_scan(self, scan_type):
        """Starts a full grid scan for the given scan_type."""
        msg = f"This will run many simulations for the {scan_type.replace('_', ' ')} plane. Continue?"
        if not messagebox.askokcancel("Start Scan?", msg): return

        for btn in self.scan_buttons.values(): btn.config(state=tk.DISABLED)
        self.clear_plot()
        
        # Setup the plot axes and keys for the specific scan
        if scan_type == 'J_vs_h':
            self.data_queue.put(('setup_plot', {
                'xtitle': 'J (Coupling)', 'ytitle': 'h (Transverse Field)',
                'xrange': (self.param_specs['J']['from_'], self.param_specs['J']['to']),
                'yrange': (self.param_specs['h']['from_'], self.param_specs['h']['to']),
                'x_key': 'J', 'y_key': 'h'
            }))
        elif scan_type == 'z_vs_theta':
            self.data_queue.put(('setup_plot', {
                'xtitle': 'Z (Coordination)', 'ytitle': 'Theta (Angle)',
                'xrange': (self.param_specs['z']['from_'], self.param_specs['z']['to']),
                'yrange': (self.param_specs['theta']['from_'], self.param_specs['theta']['to']),
                'x_key': 'z', 'y_key': 'theta'
            }))

        fixed_params = {name: var.get() for name, var in self.param_vars.items()}
        # Ensure parameters that must be integers are cast correctly
        for key in ['z', 't', 'n_qubits']:
            if key in fixed_params: fixed_params[key] = int(fixed_params[key])
        
        threading.Thread(target=run_scan_worker, args=(scan_type, fixed_params, self.sim_result_queue), daemon=True).start()

    def check_sim_queue(self):
        try:
            result_type, data = self.sim_result_queue.get_nowait()
            if result_type == 'success':
                params, result_data = data
                self.data_queue.put(('add', (params, result_data)))
            elif result_type == 'error':
                self.status_var.set("Error: See message box.")
                for btn in self.scan_buttons.values(): btn.config(state=tk.NORMAL)
                messagebox.showerror("Simulation Error", data)
            elif result_type == 'status':
                self.status_var.set(data)
                if "Scan complete" in data:
                    for btn in self.scan_buttons.values(): btn.config(state=tk.NORMAL)
                    self.data_queue.put(('scan_complete', None))
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_sim_queue)

    def clear_plot(self):
        self.data_queue.put(('clear', None))
        self.status_var.set("Plot cleared. Ready.")

    def on_closing(self):
        print("Closing application...")
        self.data_queue.put(('close', None))
        self.root.destroy()

# --- Main Execution ---
if __name__ == '__main__':
    mp.freeze_support()

    param_specs = {
        'J': {'from_': -2.0, 'to': 2.0, 'resolution': 0.1, 'default': -1.0},
        'h': {'from_': -4.0, 'to': 4.0, 'resolution': 0.1, 'default': 2.0},
        'z': {'from_': -8, 'to': 8, 'resolution': 1, 'default': 4},
        'theta': {'from_': -np.pi / 2, 'to': np.pi / 2, 'resolution': 0.01, 'default': 0.1745},
        't': {'from_': 1, 'to': 50, 'resolution': 1, 'default': 20},
        'n_qubits': {'from_': 4, 'to': 960, 'resolution': 1, 'default': 56},
    }
    
    data_queue = mp.Queue()

    plot_process = mp.Process(target=vedo_plotter_process, args=(data_queue,), daemon=True)
    plot_process.start()

    root = tk.Tk()
    app = ControlPanel(root, param_specs, data_queue)
    root.mainloop()

    plot_process.join(timeout=2)
    if plot_process.is_alive():
        print("Terminating plotter process...")
        plot_process.terminate()
        
    print("Application closed.")


# -*- coding: utf-8 -*-
"""
An interactive 3D visualizer for the Transverse Field Ising Model (TFIM)
simulation output from iterations_cli.py.

This version uses multiprocessing to run the vedo plotter in a separate
process and a multiprocessing pool to run grid scans in parallel,
ensuring a fully interactive and fast experience.
"""
import subprocess
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from vedo import Plotter, Points, Text2D, ScalarBar
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
    command = [sys.executable, 'magnetisation-iterations_cli.py']
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
        return ('error', "Error: 'magnetisation-iterations_cli.py' not found.")
    except subprocess.CalledProcessError as e:
        return ('error', f"Simulation failed with error:\n{e.stderr}")
    except Exception as e:
        return ('error', f"An unexpected error occurred: {e}")

def run_simulation_thread(params, result_queue):
    """
    Thread target for running a single simulation.
    """
    result_type, data = _execute_simulation(params)
    result_queue.put((result_type, data))

def run_full_scan_worker(fixed_params, result_queue):
    """
    Thread worker to scan the full J-h plane IN PARALLEL.
    This function prepares the jobs and hands them to a multiprocessing Pool.
    """
    j_range = np.arange(-2.0, 2.0 + 0.1, 0.1)
    h_range = np.arange(-4.0, 4.0 + 0.1, 0.1)

    all_params = []
    for h_val in h_range:
        for j_val in j_range:
            current_params = fixed_params.copy()
            current_params['J'] = round(j_val, 2)
            current_params['h'] = round(h_val, 2)
            all_params.append(current_params)
    
    total_points = len(all_params)
    count = 0

    try:
        # --- Use a multiplier for the number of processes ---
        multiplier = 3  # Use 2 times the number of CPU cores
        num_processes = mp.cpu_count() * multiplier
        print(f"Initializing pool with {num_processes} worker processes...")
        # ---

        with mp.Pool(processes=num_processes) as pool:
            # imap_unordered processes tasks in parallel and yields results as they complete
            for result_type, data in pool.imap_unordered(_execute_simulation, all_params):
                count += 1
                status_msg = f"Scanning... {count}/{total_points}"
                result_queue.put(('status', status_msg))
                result_queue.put((result_type, data))

        result_queue.put(('status', f"Scan complete. Plotted {total_points} points."))
    except Exception as e:
        error_msg = f"A multiprocessing error occurred: {e}"
        print(error_msg)
        result_queue.put(('error', error_msg))

# --- Vedo Plotter Process ---
def vedo_plotter_process(data_queue, param_specs):
    """
    This function runs in a separate process to handle all vedo plotting.
    """
    j_range = (param_specs['J']['from_'], param_specs['J']['to'])
    h_range = (param_specs['h']['from_'], param_specs['h']['to'])
    z_range = (0.0, 1.0)

    plt = Plotter(
        axes=dict(
            xtitle='J (Coupling)', ytitle='h (Transverse Field)', ztitle='Avg. Measured Value',
            zlabel_size=0.02, ylabel_size=0.02, xlabel_size=0.02,
            xrange=j_range, yrange=h_range, zrange=z_range
        ),
        bg='black',
        title="TFIM Phase Visualizer"
    )

    plotted_data = []
    point_cloud_actor = None
    info_text_actor = Text2D("Initializing...", pos='top-left', s=1.1, bg='gray', alpha=0.7)
    scalar_bar_actor = None
    
    plt.show(info_text_actor, interactive=False)

    def update_plot(timer_id):
        nonlocal point_cloud_actor, info_text_actor, scalar_bar_actor
        try:
            while not data_queue.empty():
                command, data = data_queue.get_nowait()
                
                if command == 'add':
                    params, (avg_z, time_str) = data
                    new_point = (params['J'], params['h'], avg_z)
                    plotted_data.append(new_point)
                    points_array = np.array(plotted_data)

                    plt.remove(point_cloud_actor, info_text_actor, scalar_bar_actor)

                    z_values = points_array[:, 2]
                    vmin, vmax = 0.0, 1.0
                    point_cloud_actor = Points(points_array, r=12).cmap('viridis', z_values, vmin=vmin, vmax=vmax)

                    scalar_bar_actor = ScalarBar(
                        point_cloud_actor, title="Avg. Measured Value", pos=((0.85, 0.4), (0.9, 0.9))
                    )

                    info_text = (f"Plotted points: {len(plotted_data)}\n"
                                 f"Last point: J={params['J']:.2f}, h={params['h']:.2f}\n"
                                 f"Avg. Value: {avg_z:.4f}\n"
                                 f"{time_str}")
                    info_text_actor = Text2D(info_text, pos='top-left', s=1.1, bg='yellow', alpha=0.7)

                    plt.add(point_cloud_actor, info_text_actor, scalar_bar_actor)

                elif command == 'clear':
                    plotted_data.clear()
                    plt.remove(point_cloud_actor, info_text_actor, scalar_bar_actor)
                    point_cloud_actor = None
                    scalar_bar_actor = None
                    info_text_actor = Text2D("Plot cleared.", pos='top-left', s=1.2, bg='gray', alpha=0.7)
                    plt.add(info_text_actor)
                    plt.reset_camera()

                elif command == 'close':
                    plt.close()
                    return

        except Exception as e:
            print(f"Error in vedo process: {e}")
        
        plt.render()

    plt.add_callback('timer', update_plot)
    plt.timer_callback('create', dt=100)
    plt.interactive().close()

# --- GUI Class ---
class ControlPanel:
    """
    Manages the tkinter GUI, and sends data to the vedo process.
    """
    def __init__(self, root, param_specs, data_queue):
        self.root = root
        self.param_specs = param_specs
        self.data_queue = data_queue
        self.sim_result_queue = queue.Queue()
        self.param_vars = {}
        self.scan_button = None

        self.root.title("TFIM Control Panel")
        self.root.geometry("400x650")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready. Move a slider or start a scan.")
        self._create_controls(main_frame)
        
        self.root.after(100, self.check_sim_queue)

    def _create_controls(self, parent):
        slider_frame = ttk.Frame(parent)
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

            slider = ttk.Scale(
                frame, from_=spec['from_'], to=spec['to'],
                orient=tk.HORIZONTAL, variable=var, command=command
            )
            slider.pack(fill=tk.X, expand=True)
            slider.bind("<ButtonRelease-1>", self.run_simulation_from_sliders)

        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=20)

        self.scan_button = ttk.Button(button_frame, text="Run Full Scan (J vs h)", command=self.start_full_scan)
        self.scan_button.pack(side=tk.LEFT, expand=True, padx=5)
        
        clear_button = ttk.Button(button_frame, text="Clear Plot", command=self.clear_plot)
        clear_button.pack(side=tk.RIGHT, expand=True, padx=5)

        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def run_simulation_from_sliders(self, event=None):
        current_params = {name: var.get() for name, var in self.param_vars.items()}
        for key in ['z', 't', 'n_qubits']:
            if key in current_params:
                current_params[key] = int(current_params[key])

        self.status_var.set("Running single simulation...")
        thread = threading.Thread(target=run_simulation_thread, args=(current_params, self.sim_result_queue), daemon=True)
        thread.start()
        
    def start_full_scan(self):
        """Starts the full J-h grid scan in a new thread."""
        if messagebox.askokcancel("Start Scan?", "This will run many simulations in parallel. Continue?"):
            self.scan_button.config(state=tk.DISABLED)
            self.clear_plot()
            
            fixed_params = {name: var.get() for name, var in self.param_vars.items()}
            fixed_params.pop('J', None)
            fixed_params.pop('h', None)
            for key in ['z', 't', 'n_qubits']:
                if key in fixed_params:
                    fixed_params[key] = int(fixed_params[key])
            
            thread = threading.Thread(target=run_full_scan_worker, args=(fixed_params, self.sim_result_queue), daemon=True)
            thread.start()

    def check_sim_queue(self):
        try:
            result_type, data = self.sim_result_queue.get_nowait()
            if result_type == 'success':
                params, result_data = data
                self.status_var.set(f"Plotting: J={params['J']:.2f}, h={params['h']:.2f}")
                self.data_queue.put(('add', (params, result_data)))

            elif result_type == 'error':
                self.status_var.set("Error: See message box for details.")
                self.scan_button.config(state=tk.NORMAL)
                messagebox.showerror("Simulation Error", data)
            
            elif result_type == 'status':
                self.status_var.set(data)
                if "Scan complete" in data:
                    self.scan_button.config(state=tk.NORMAL)
                    
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
    # This is important for multiprocessing on all platforms
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

    plot_process = mp.Process(target=vedo_plotter_process, args=(data_queue, param_specs))
    plot_process.start()

    root = tk.Tk()
    app = ControlPanel(root, param_specs, data_queue)
    root.mainloop()

    plot_process.join()
    print("Application closed.")

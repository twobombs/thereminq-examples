# -*- coding: utf-8 -*-
"""
An interactive 3D visualizer for the Transverse Field Ising Model (TFIM)
simulation output from iterations_cli.py.

This version uses multiprocessing to run the vedo plotter in a separate
process and a multiprocessing pool to run grid scans in parallel,
ensuring a fully interactive and fast experience.

NOTE: This version plots a point for EACH measurement shot. A single simulation
run with 1000 shots will add 1000 points to the plot. A full scan can
generate millions of points and may be slow or memory-intensive.
Consider reducing the 'shots' parameter for full scans.

To run:
1. Save this file as 'interactive_visualizer.py'.
2. Save the accompanying 'magnetisation_iterations_cli.py' in the same directory.
3. Ensure you have the required libraries: pip install numpy vedo PyQrackIsing
4. Run this script: python interactive_visualizer.py
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
import time

# --- Simulation Runners ---

def _execute_simulation(params):
    """
    Core function to execute the CLI script and parse the output.
    This is a top-level function so it can be 'pickled' by multiprocessing.
    Returns a tuple: (result_type, data)
    """
    # Ensure the target script is found, regardless of the working directory
    try:
        # In a frozen app (e.g., PyInstaller), sys.executable is the path to the app
        # In a script, it's the path to the python interpreter.
        if getattr(sys, 'frozen', False):
            # The script is running in a bundle
            base_path = sys._MEIPASS
        else:
            # The script is running in a normal Python environment
            base_path = '.'
        
        cli_script_path = f"{base_path}/iterations_cli.py"
        command = [sys.executable, cli_script_path]

    except Exception:
        # Fallback for safety
        command = [sys.executable, 'iterations_cli.py']

    for key, value in params.items():
        command.extend([f'--{key}', str(value)])

    try:
        # Platform-specific flags to prevent console windows from popping up
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(
            command, capture_output=True, text=True, check=True, creationflags=creationflags
        )

        output_lines = result.stdout.strip().split('\n')
        samples_line = ""
        time_line = "Calculation took: N/A"

        # Safely parse the output from the CLI script
        try:
            samples_index = output_lines.index("## Output Samples (Decimal Comma-Separated) ##")
            samples_line = output_lines[samples_index + 1]
        except (ValueError, IndexError):
            pass # samples_line will remain empty

        for line in output_lines:
            if line.startswith("Calculation took:"):
                time_line = line
                break

        if not samples_line:
            return ('error', "Could not find sample output in simulation results.")

        # --- UPDATED PARSING LOGIC ---
        # The CLI script returns integer bitmasks representing the state of the qubits for each shot.
        # We calculate the magnetization for each shot and return the full list.
        try:
            int_samples = [int(s) for s in samples_line.split(',')]
            if not int_samples:
                 return ('error', "Simulation returned empty sample list.")

            n_qubits = params.get('n_qubits')
            if n_qubits is None:
                return ('error', "'n_qubits' parameter not found, cannot calculate magnetization.")

            # For each sample (shot), calculate its magnetization.
            # Magnetization = (spins_up - spins_down) / total_spins
            # spins_up = number of set bits (1s)
            # spins_down = n_qubits - spins_up
            # Magnetization = (spins_up - (n_qubits - spins_up)) / n_qubits
            #               = (2 * spins_up - n_qubits) / n_qubits
            magnetization_per_shot = []
            for sample in int_samples:
                # Count the number of set bits (spin up) in the integer sample.
                # int.bit_count() is Python 3.10+, using bin().count() for better compatibility.
                spins_up = bin(sample).count('1')
                shot_magnetization = (2 * spins_up - n_qubits) / n_qubits
                magnetization_per_shot.append(shot_magnetization)

            # Return the entire list of magnetization values for each shot.
            return ('success', (params, (magnetization_per_shot, time_line)))

        except (ValueError, TypeError) as e:
            return ('error', f"Could not parse simulation output samples: {e}\nOutput line: '{samples_line}'")


    except FileNotFoundError:
        return ('error', "Error: 'magnetisation_iterations_cli.py' not found. Ensure it's in the same directory.")
    except subprocess.CalledProcessError as e:
        # Include stderr from the child process for better debugging
        return ('error', f"Simulation script failed with error:\n{e.stderr}")
    except Exception as e:
        return ('error', f"An unexpected error occurred: {e}")

def run_simulation_thread(params, result_queue):
    """
    Thread target for running a single simulation. Puts the result in a queue.
    """
    result_type, data = _execute_simulation(params)
    result_queue.put((result_type, data))

def run_full_scan_worker(fixed_params, result_queue):
    """
    Worker to scan the full J-h plane in parallel using a multiprocessing Pool.
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
        # Using a multiplier for the number of processes can be beneficial for
        # I/O-bound or mixed tasks, as it allows the CPU to switch to other
        # tasks while one is waiting.
        multiplier = 3
        num_processes = max(1, mp.cpu_count() * multiplier)
        result_queue.put(('status', f"Initializing pool with {num_processes} workers..."))

        with mp.Pool(processes=num_processes) as pool:
            # imap_unordered is efficient as it processes tasks in parallel
            # and yields results as soon as they complete.
            for result_type, data in pool.imap_unordered(_execute_simulation, all_params):
                count += 1
                # Update status for the GUI
                status_msg = f"Scanning... {count}/{total_points}"
                result_queue.put(('status', status_msg))
                # Send the actual data for plotting
                result_queue.put((result_type, data))

        result_queue.put(('status', f"Scan complete. Plotted {total_points} parameter sets."))
    except Exception as e:
        error_msg = f"A multiprocessing error occurred: {e}"
        print(error_msg, file=sys.stderr)
        result_queue.put(('error', error_msg))


# --- Vedo Plotter Process ---

def vedo_plotter_process(data_queue, param_specs):
    """
    This function runs in a separate process to handle all vedo plotting,
    preventing the GUI from freezing during rendering.
    """
    j_range = (param_specs['J']['from_'], param_specs['J']['to'])
    h_range = (param_specs['h']['from_'], param_specs['h']['to'])
    # The measured value can be negative, so we adjust the z_range
    z_range = (-1.0, 1.0) 

    plt = Plotter(
        axes=dict(
            xtitle='J (Coupling)', ytitle='h (Transverse Field)', ztitle='Shot Magnetization',
            zlabel_size=0.02, ylabel_size=0.02, xlabel_size=0.02,
            xrange=j_range, yrange=h_range, zrange=z_range
        ),
        bg='black',
        title="TFIM Phase Visualizer"
    )

    # State variables for the plot
    plotted_data = []
    point_cloud_actor = None
    info_text_actor = Text2D("Initializing...", pos='top-left', s=1.1, bg='gray', alpha=0.7)
    scalar_bar_actor = None
    
    plt.show(info_text_actor, interactive=False)

    def update_plot(timer_id):
        """Callback function to update the plot with new data from the queue."""
        nonlocal point_cloud_actor, info_text_actor, scalar_bar_actor
        try:
            # Process all pending items in the queue
            while not data_queue.empty():
                command, data = data_queue.get_nowait()
                
                if command == 'add':
                    # --- ROBUST UPDATE LOGIC ---
                    # This logic forces a full refresh of the plot actors on each update
                    # to prevent rendering glitches where new data is not displayed.
                    
                    params, (z_values_list, time_str) = data
                    
                    new_points = [(params['J'], params['h'], z) for z in z_values_list]
                    plotted_data.extend(new_points)
                    points_array = np.array(plotted_data)
                    z_values = points_array[:, 2]
                    vmin, vmax = -1.0, 1.0

                    # Remove all previous actors to ensure a clean slate.
                    actors_to_remove = [point_cloud_actor, scalar_bar_actor, info_text_actor]
                    plt.remove([a for a in actors_to_remove if a is not None])

                    # Recreate all actors from the complete dataset.
                    point_cloud_actor = Points(points_array, r=4).cmap('viridis', z_values, vmin=vmin, vmax=vmax)
                    
                    scalar_bar_actor = ScalarBar(
                        point_cloud_actor, title="Shot Magnetization", pos=((0.85, 0.3), (0.9, 0.7))
                    )
                    
                    info_text = (f"Total points: {len(plotted_data)}\n"
                                 f"Last params: J={params['J']:.2f}, h={params['h']:.2f}\n"
                                 f"Points added: {len(z_values_list)}\n"
                                 f"{time_str}")
                    info_text_actor = Text2D(info_text, pos='top-left', s=1.1, bg='yellow', alpha=0.7)

                    # Add the newly created actors back to the scene.
                    plt.add(point_cloud_actor, scalar_bar_actor, info_text_actor)
                    
                    # Reset camera to frame the updated scene.
                    if len(plotted_data) > 0:
                        plt.reset_camera()


                elif command == 'clear':
                    plotted_data.clear()
                    # Remove all actors completely on clear
                    actors_to_remove = [point_cloud_actor, scalar_bar_actor, info_text_actor]
                    plt.remove([a for a in actors_to_remove if a is not None])
                    point_cloud_actor = None
                    scalar_bar_actor = None
                    info_text_actor = Text2D("Plot cleared.", pos='top-left', s=1.2, bg='gray', alpha=0.7)
                    plt.add(info_text_actor)
                    plt.reset_camera()

                elif command == 'close':
                    plt.close()
                    return # Exit the update loop

        except queue.Empty:
            pass # No new data, just continue
        except Exception as e:
            print(f"Error in vedo process: {e}", file=sys.stderr)
        
        plt.render()

    # Register the update function as a timer callback
    plt.add_callback('timer', update_plot)
    plt.timer_callback('create', dt=100) # Check for new data every 100ms
    plt.interactive().close() # Start interactive mode and close when done


# --- GUI Class ---

class ControlPanel:
    """
    Manages the tkinter GUI, user inputs, and communication with the plotter process.
    """
    def __init__(self, root, param_specs, data_queue):
        self.root = root
        self.param_specs = param_specs
        self.data_queue = data_queue
        self.sim_result_queue = queue.Queue() # For results from simulation threads
        self.param_vars = {}
        self.scan_button = None

        self.root.title("TFIM Control Panel")
        self.root.geometry("400x700") # Increased height for the new slider
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready. Move a slider or start a scan.")
        self._create_controls(main_frame)
        
        # Start checking for simulation results
        self.root.after(100, self.check_sim_queue)

    def _create_controls(self, parent):
        """Creates all the GUI widgets."""
        slider_frame = ttk.LabelFrame(parent, text="Simulation Parameters")
        slider_frame.pack(fill=tk.X, expand=True, pady=10)
        
        for name, spec in self.param_specs.items():
            frame = ttk.Frame(slider_frame)
            frame.pack(fill=tk.X, pady=8, padx=10)
            
            label = ttk.Label(frame, text=f"{name.replace('_', ' ').title()}:", width=12, anchor=tk.W)
            label.pack(side=tk.LEFT)

            var = tk.DoubleVar(value=spec['default'])
            self.param_vars[name] = var

            value_label = ttk.Label(frame, text=f"{var.get():.2f}", width=6)
            value_label.pack(side=tk.RIGHT, padx=(5, 0))

            # Use a lambda to capture the correct variables for the callback
            command = lambda val, l=value_label, v=var, s=spec: \
                l.config(text=f"{v.get():.2f}" if s.get('resolution', 0.1) < 1 else f"{int(v.get())}")

            slider = ttk.Scale(
                frame, from_=spec['from_'], to=spec['to'],
                orient=tk.HORIZONTAL, variable=var, command=command
            )
            slider.pack(fill=tk.X, expand=True)
            # Run simulation only when the user releases the mouse button
            slider.bind("<ButtonRelease-1>", self.run_simulation_from_sliders)

        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=20)

        self.scan_button = ttk.Button(button_frame, text="Run Full Scan (J vs h)", command=self.start_full_scan)
        self.scan_button.pack(side=tk.LEFT, expand=True, padx=5, ipady=5)
        
        clear_button = ttk.Button(button_frame, text="Clear Plot", command=self.clear_plot)
        clear_button.pack(side=tk.RIGHT, expand=True, padx=5, ipady=5)

        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def run_simulation_from_sliders(self, event=None):
        """Runs a single simulation based on current slider values."""
        current_params = {name: var.get() for name, var in self.param_vars.items()}
        # Ensure integer parameters are correctly formatted
        for key in ['z', 't', 'n_qubits', 'shots']:
            if key in current_params:
                current_params[key] = int(current_params[key])

        self.status_var.set("Running single simulation...")
        # Run in a separate thread to not block the GUI
        thread = threading.Thread(target=run_simulation_thread, args=(current_params, self.sim_result_queue), daemon=True)
        thread.start()
        
    def start_full_scan(self):
        """Starts the full J-h grid scan in a new thread."""
        if messagebox.askokcancel("Start Scan?", "This will run many simulations in parallel and may be CPU-intensive. Continue?"):
            self.scan_button.config(state=tk.DISABLED)
            self.clear_plot()
            
            # Get fixed parameters from sliders, excluding J and h which will be scanned
            fixed_params = {name: var.get() for name, var in self.param_vars.items()}
            fixed_params.pop('J', None)
            fixed_params.pop('h', None)
            for key in ['z', 't', 'n_qubits', 'shots']:
                if key in fixed_params:
                    fixed_params[key] = int(fixed_params[key])
            
            # Run the scan worker in a thread to manage the multiprocessing pool
            thread = threading.Thread(target=run_full_scan_worker, args=(fixed_params, self.sim_result_queue), daemon=True)
            thread.start()

    def check_sim_queue(self):
        """Periodically checks the queue for results from simulation threads."""
        try:
            result_type, data = self.sim_result_queue.get_nowait()
            
            if result_type == 'success':
                params, result_data = data
                self.status_var.set(f"Plotting {len(result_data[0])} points for J={params.get('J', 0):.2f}, h={params.get('h', 0):.2f}")
                # --- DEADLOCK PREVENTION ---
                # The vedo process can be slow to render when there are many points.
                # If its data queue is full, a blocking put() would halt the GUI
                # and the entire simulation pipeline.
                # By using put_nowait(), we send the data if there's room, but if not,
                # we simply drop this data frame. This keeps the simulation running,
                # and the plot will appear to update in larger jumps instead of halting.
                try:
                    self.data_queue.put_nowait(('add', (params, result_data)))
                except queue.Full:
                    pass # Plotter is busy, drop frame and continue.

            elif result_type == 'error':
                self.status_var.set("Error: See message box for details.")
                self.scan_button.config(state=tk.NORMAL) # Re-enable scan button on error
                messagebox.showerror("Simulation Error", data)
            
            elif result_type == 'status':
                self.status_var.set(data)
                if "Scan complete" in data:
                    self.scan_button.config(state=tk.NORMAL)
                    
        except queue.Empty:
            # Queue is empty, nothing to do
            pass
        finally:
            # Schedule the next check
            self.root.after(100, self.check_sim_queue)

    def clear_plot(self):
        """Sends a command to clear the vedo plot."""
        self.data_queue.put(('clear', None))
        self.status_var.set("Plot cleared. Ready.")

    def on_closing(self):
        """Handles the application closing event."""
        print("Closing application...")
        self.data_queue.put(('close', None)) # Tell the plotter process to close
        self.root.destroy()


# --- Main Execution ---

if __name__ == '__main__':
    # This is crucial for multiprocessing to work correctly on all platforms,
    # especially Windows and macOS.
    mp.freeze_support()

    # Define the parameters, their ranges, and default values for the GUI
    param_specs = {
        'J': {'from_': -2.0, 'to': 2.0, 'resolution': 0.1, 'default': -1.0},
        'h': {'from_': -4.0, 'to': 4.0, 'resolution': 0.1, 'default': 2.0},
        'z': {'from_': -8, 'to': 8, 'resolution': 1, 'default': 4},
        'theta': {'from_': -np.pi / 2, 'to': np.pi / 2, 'resolution': 0.01, 'default': 0.1745},
        't': {'from_': 1, 'to': 50, 'resolution': 1, 'default': 20},
        'n_qubits': {'from_': 4, 'to': 960, 'resolution': 1, 'default': 56},
        'shots': {'from_': 100, 'to': 5000, 'resolution': 1, 'default': 100},
    }
    
    # Create a queue for communication between the GUI and the plotter process
    data_queue = mp.Queue()

    # Start the plotter in a separate process
    plot_process = mp.Process(target=vedo_plotter_process, args=(data_queue, param_specs))
    plot_process.start()

    # Create and run the main tkinter GUI
    root = tk.Tk()
    app = ControlPanel(root, param_specs, data_queue)
    root.mainloop()

    # Wait for the plotter process to finish before exiting
    plot_process.join()
    print("Application closed.")


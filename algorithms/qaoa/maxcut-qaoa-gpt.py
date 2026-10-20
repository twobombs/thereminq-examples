# source:
# https://medium.com/quantum-engineering/qaoa-gpt-how-ai-generates-quantum-circuits-for-optimization-7f8f6a4d800a
# todo: explore optimisations for other areas

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import quantum_info as qi
from scipy.optimize import minimize
from scipy import sparse
import math
import warnings
from networkx import Graph

##########################
### Core QAOA Functions ##
##########################

def initial_density_matrix(no_qubits):
    """Returns density matrix for QAOA initial state."""
    dim = 2**no_qubits
    dens_mat = qi.DensityMatrix(np.full((dim, dim), 1/dim))
    return sparse.csr_matrix(dens_mat.data)

def cut_hamiltonian(graph):
    """Returns cut Hamiltonian operator."""
    if not isinstance(graph, Graph):
        raise Exception("Graph must be instance of networkx Graph class!")

    no_nodes = graph.number_of_nodes()
    no_ops = graph.number_of_edges()
    pauli_strings = [None] * no_ops
    coeffs = [None] * no_ops
    index = 0

    for i in range(no_nodes):
        for k in range(i+1, no_nodes):
            if graph.get_edge_data(i, k) is not None:
                tmp_str = 'I' * i + 'Z' + 'I' * (k-i-1) + 'Z' + 'I' * (no_nodes - k - 1)
                tmp_str = tmp_str[::-1]
                pauli_strings[index] = tmp_str
                coeffs[index] = (-0.5) * graph.get_edge_data(i, k)['weight']
                index += 1

    return sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_operator().data)

def cut_unitary(graph, parameter, dict_paulis):
    """Returns unitary operator for cut Hamiltonian exponential."""
    if not isinstance(graph, nx.Graph):
        raise Exception("Invalid graph type")

    first = True
    for edge in graph.edges:
        weight = graph.get_edge_data(*edge)['weight']
        total_param = 0.5 * parameter * weight
        key = f"Z{edge[0]}Z{edge[1]}"
        key = key if key in dict_paulis else f"Z{edge[1]}Z{edge[0]}"
        
        if key not in dict_paulis:
            raise Exception("Pauli string not found")
            
        tmp_matrix = dict_paulis['I'] * math.cos(total_param) + dict_paulis[key] * math.sin(total_param) * 1j
        result = tmp_matrix if first else tmp_matrix * result
        first = False

    return result

def mixer_unitary(mixer_type, parameter_value, dict_paulis, no_nodes):
    """Returns mixer unitary operator."""
    if mixer_type in ['standard_x', 'standard_y']:
        pauli = mixer_type[-1].upper()
        result = None
        for i in range(no_nodes):
            term = math.cos(parameter_value) * dict_paulis['I'] - 1j * math.sin(parameter_value) * dict_paulis[f"{pauli}{i}"]
            result = term if result is None else result * term
    else:
        result = math.cos(parameter_value) * dict_paulis['I'] - 1j * math.sin(parameter_value) * dict_paulis[mixer_type]
    
    return result

##########################
### Ansatz Construction ##
##########################

def build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_dict, 
                          ham_layers=None, noisy=False, noise_prob=0.0):
    """Builds ADAPT-QAOA ansatz circuit."""
    if not isinstance(graph, Graph):
        raise Exception("Graph must be instance of Networkx Graph class!")

    no_layers = len(mixer_params)
    if no_layers != len(mixer_list):
        raise Exception('Incompatible number of mixer types and parameters!')
    
    ham_layers = ham_layers or list(range(1, no_layers+1))
    if len(ham_params) != len(ham_layers):
        raise Exception("Incompatible number of Hamiltonian parameters and layers!")
    
    no_qubits = graph.number_of_nodes()
    dens_mat = initial_density_matrix(no_qubits)
    ham_unitaries_count = 0
    
    for layer in range(no_layers):
        if len(ham_layers) > ham_unitaries_count and ham_layers[ham_unitaries_count] == layer + 1:
            cut_unit = cut_unitary(graph, ham_params[ham_unitaries_count], pauli_dict)
            dens_mat = (cut_unit * dens_mat) * (cut_unit.conj().T)
            ham_unitaries_count += 1
            if noisy:
                dens_mat = noisy_ham_unitary_evolution(dens_mat, noise_prob, graph, pauli_dict)

        mix_unit = mixer_unitary(mixer_list[layer], mixer_params[layer], pauli_dict, no_qubits)
        dens_mat = (mix_unit * dens_mat) * (mix_unit.conj().T)
        if noisy:
            dens_mat = noisy_mixer_unitary_evolution(dens_mat, noise_prob, mixer_list[layer], pauli_dict)

    return dens_mat

def build_standard_qaoa_ansatz(graph, parameter_list, pauli_dict, noisy=False, noise_prob=0.0):
    """Builds standard QAOA ansatz circuit."""
    if not isinstance(graph, Graph):
        raise Exception("Graph must be instance of Networkx Graph class!")

    no_layers = len(parameter_list) // 2
    ham_parameters = parameter_list[:no_layers]
    mixer_parameters = parameter_list[no_layers:]
    no_qubits = graph.number_of_nodes()
    dens_mat = initial_density_matrix(no_qubits)
    
    for layer in range(no_layers):
        cut_unit = cut_unitary(graph, ham_parameters[layer], pauli_dict)
        dens_mat = (cut_unit * dens_mat) * (cut_unit.conj().T)
        if noisy:
            dens_mat = noisy_ham_unitary_evolution(dens_mat, noise_prob, graph, pauli_dict)
    
        mix_unit = mixer_unitary('standard_x', mixer_parameters[layer], pauli_dict, no_qubits)
        dens_mat = (mix_unit * dens_mat) * (mix_unit.conj().T)

    return dens_mat

##########################
### Mixer Construction ###
##########################

def build_all_mixers(graph):
    """Builds all possible mixers for the given graph."""
    if not isinstance(graph, Graph):
        raise Exception("Graph must be instance of Networkx Graph class!")

    dict_mixers = {}
    no_qubits = graph.number_of_nodes()
    single_qubit_mixers = ["X", "Y"]
    double_qubit_mixers = ["XZ", "YZ", "XY", "XX", "YY"]

    # Single qubit mixers
    for mixer_type in single_qubit_mixers:
        for qubit in range(no_qubits):
            key = f"{mixer_type}{qubit}"
            dict_mixers[key] = X_mixer(graph, qubit) if mixer_type == 'X' else Y_mixer(graph, qubit)

    # Double qubit mixers
    for mixer_type in double_qubit_mixers:
        for qubit_1 in range(no_qubits):
            for qubit_2 in range(no_qubits):
                if qubit_1 == qubit_2:
                    continue
                key = f"{mixer_type[0]}{qubit_1}{mixer_type[1]}{qubit_2}"
                
                if mixer_type == 'XZ':
                    dict_mixers[key] = XZ_mixer(graph, qubit_1, qubit_2)
                elif mixer_type == 'YZ':
                    dict_mixers[key] = YZ_mixer(graph, qubit_1, qubit_2)
                elif mixer_type == 'XY':
                    dict_mixers[key] = XY_mixer(graph, qubit_1, qubit_2)
                elif qubit_2 > qubit_1:
                    if mixer_type == 'XX':
                        dict_mixers[key] = XX_mixer(graph, qubit_1, qubit_2)
                    elif mixer_type == 'YY':
                        dict_mixers[key] = YY_mixer(graph, qubit_1, qubit_2)

    return dict_mixers

def build_all_paulis(no_nodes):
    """Builds all Pauli matrices for unitary building blocks."""
    result = {}
    mixer_types = ['X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XZ', 'YZ', 'XY']

    for mixer in mixer_types:
        if len(mixer) == 1:
            for node in range(no_nodes):
                key = f"{mixer}{node}"
                pauli_string = 'I' * node + mixer + 'I' * (no_nodes-node-1)
                result[key] = sparse.csr_matrix(qi.Pauli(pauli_string[::-1]).to_matrix())
        elif len(mixer) == 2:
            for node_1 in range(no_nodes):
                for node_2 in range(no_nodes):
                    if node_1 == node_2 or (mixer[0] == mixer[1] and node_2 < node_1):
                        continue
                    
                    key = f"{mixer[0]}{node_1}{mixer[1]}{node_2}"
                    if node_1 > node_2:
                        larger_node, larger_type = node_1, mixer[0]
                        smaller_node, smaller_type = node_2, mixer[1]
                    else:
                        larger_node, larger_type = node_2, mixer[1]
                        smaller_node, smaller_type = node_1, mixer[0]
                        
                    pauli_string = ('I' * smaller_node + smaller_type + 
                                   'I' * (larger_node-smaller_node-1) + 
                                   larger_type + 'I' * (no_nodes - larger_node - 1))
                    result[key] = sparse.csr_matrix(qi.Pauli(pauli_string[::-1]).to_matrix())

    result['I'] = sparse.csr_matrix(np.identity(2**no_nodes, dtype=complex))
    return result

##########################
### Optimization Tools ###
##########################

def find_mixer_gradients(dens_mat, mixer_dict, pauli_dict, graph, 
                        apply_ham_unitary=True, gamma_0=0.01, noisy=False, noise_prob=0.0):
    """Finds gradients for all mixer operators."""
    if apply_ham_unitary and gamma_0 != 0.0:
        cut_unit = cut_unitary(graph, gamma_0, pauli_dict)
        new_dens_mat = (cut_unit * dens_mat) * (cut_unit.conj().T)
        if noisy:
            new_dens_mat = noisy_ham_unitary_evolution(new_dens_mat, noise_prob, graph, pauli_dict)
    else:
        new_dens_mat = dens_mat

    dict_gradients = {'standard_x': 0.0, 'standard_y': 0.0}

    for mixer_type, mixer in mixer_dict.items():
        gradient = mixer.find_exact_gradient(new_dens_mat)
        
        # Check if single-qubit Pauli
        if sum(1 for letter in mixer_type if letter in ['X', 'Y', 'Z']) == 1:
            if mixer_type[0] == 'X':
                dict_gradients['standard_x'] += gradient
            elif mixer_type[0] == 'Y':
                dict_gradients['standard_y'] += gradient

        dict_gradients[mixer_type] = abs(gradient)

    dict_gradients['standard_x'] = abs(dict_gradients['standard_x'])
    dict_gradients['standard_y'] = abs(dict_gradients['standard_y'])

    return sorted(dict_gradients.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

def find_optimal_cut(graph):
    """Finds optimal cut using exhaustive search."""
    solution = []
    offset = nx.adjacency_matrix(graph).sum() / 4
    
    for i in range(2**graph.number_of_nodes()):
        bitstring = bin(i)[2:].zfill(graph.number_of_nodes())
        cut = evaluate_cut(graph, bitstring)
        
        if not solution or solution[1] < cut:
            solution = [bitstring, cut, cut - offset]

    return solution

def evaluate_cut(graph, bitstring):
    """Evaluates cut value for given bitstring."""
    return sum(graph.get_edge_data(*edge)['weight'] 
              for edge in graph.edges 
              if bitstring[int(edge[0])] != bitstring[int(edge[1])])

##########################
### Visualization Tools ##
##########################

def plot_approximation_ratios(results):
    """Plots the approximation ratios over layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cut approximation ratio plot
    ax1.plot(results['cut_approx_ratios'], 'o-', label='Cut Approximation Ratio')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Approximation Ratio')
    ax1.set_title('Cut Approximation Ratio vs. Layers')
    ax1.grid(True)
    
    # Hamiltonian approximation ratio plot
    ax2.plot(results['ham_approx_ratios'], 'o-', color='orange', label='Hamiltonian Approximation Ratio')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Approximation Ratio')
    ax2.set_title('Hamiltonian Approximation Ratio vs. Layers')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_mixer_gradients(results):
    """Plots the mixer gradients over layers."""
    plt.figure(figsize=(10, 6))
    
    for mixer, gradients in results['all_mixers'].items():
        if len(gradients) == len(results['best_mixers']):
            plt.plot(gradients, 'o-', label=mixer)
    
    plt.xlabel('Layer')
    plt.ylabel('Gradient Magnitude')
    plt.title('Mixer Gradient Magnitudes vs. Layers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_optimal_parameters(results):
    """Plots the optimal parameters over layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mixer parameters
    ax1.plot(results['best_mixer_parameters'], 'o-', label='Mixer Parameters')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Parameter Value')
    ax1.set_title('Optimal Mixer Parameters vs. Layers')
    ax1.grid(True)
    
    # Hamiltonian parameters
    ax2.plot(results['best_ham_parameters'], 'o-', color='orange', label='Hamiltonian Parameters')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Optimal Hamiltonian Parameters vs. Layers')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

##########################
### Main Execution ###
##########################

def run_adapt_qaoa(graph, pauli_ops_dict, gradient_ops_dict, max_depth, 
                  beta_0=0.0, gamma_0=0.01, rel_gtol=1e-2, etol=-1):
    """Runs ADAPT-QAOA algorithm with visualization."""
    max_cut_solution = find_optimal_cut(graph)
    max_cut_value, max_ham_value = max_cut_solution[1], max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value
    hamiltonian = sparse.csr_matrix(cut_hamiltonian(graph))

    mixer_params, mixer_list, ham_params, ham_layers = [], [], [], []
    ham_approx_ratios, cut_approx_ratios = [], []
    all_mixers_per_layer_dict = {}

    curr_dens_mat = initial_density_matrix(graph.number_of_nodes())
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    ham_approx_ratios.append(curr_ham_estimate)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_adapt_qaoa_ansatz(
            graph, 
            parameter_values[:(no_params-no_ham_layers)], 
            mixers, 
            parameter_values[(no_params-no_ham_layers):], 
            pauli_ops_dict, 
            ham_unitary_layers
        )
        return -1.0 * (hamiltonian * dens_mat).trace().real

    for curr_layer in range(1, max_depth+1):
        print(f"Finding Best Mixer for layer {curr_layer}...")
        all_mixer_gradients = find_mixer_gradients(
            curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, 
            apply_ham_unitary=True, gamma_0=gamma_0
        )

        best_mixer, best_gradient = all_mixer_gradients[0]
        mixer_list.append(best_mixer)
        gradient_tolerance = best_gradient * rel_gtol
        print(f"\tBest mixer: {best_mixer} (gradient: {best_gradient:.4f})")

        # Store all gradients for plotting
        for mixer, gradient in all_mixer_gradients:
            if mixer in all_mixers_per_layer_dict:
                all_mixers_per_layer_dict[mixer].append(gradient)
            else:
                all_mixers_per_layer_dict[mixer] = [gradient]

        print(f"Optimising layer {curr_layer}...")
        initial_guesses = mixer_params + [beta_0] + ham_params + [gamma_0]
        ham_layers.append(curr_layer)
        
        result = minimize(
            obj_func, initial_guesses, 
            args=(mixer_list, ham_layers), 
            method="BFGS", 
            options={'gtol': gradient_tolerance}
        )

        print(f"\tOptimization completed in {result.nit} iterations")
        print(f"\tSuccess: {result.success} ({result.message})")

        parameter_list = list(result.x)
        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        
        curr_dens_mat = build_adapt_qaoa_ansatz(
            graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers
        )
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate)
        cut_approx_ratios.append((curr_ham_estimate + ham_offset) / max_cut_value)
        
        print(f"\nCurrent Cut Approximation Ratio: {cut_approx_ratios[-1]:.4f}\n")

        # Check convergence
        if (etol > 0 and len(ham_approx_ratios) > 1 and 
            abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol):
            break

    # Normalize Hamiltonian approximation ratios
    ham_approx_ratios = [x/max_ham_value for x in ham_approx_ratios]

    results = {
        'cut_approx_ratios': cut_approx_ratios,
        'ham_approx_ratios': ham_approx_ratios,
        'best_mixers': mixer_list,
        'best_mixer_parameters': mixer_params,
        'best_ham_parameters': ham_params,
        'all_mixers': all_mixers_per_layer_dict
    }

    # Plot results
    plot_approximation_ratios(results)
    plot_mixer_gradients(results)
    plot_optimal_parameters(results)

    return results

##########################
### Example Usage ###
##########################

if __name__ == "__main__":
    # Create a graph
    graph = nx.Graph()
    graph.add_edges_from([
        (0, 1, {'weight': 1}),
        (1, 2, {'weight': 1}),
        (2, 0, {'weight': 1})
    ])

    # Visualize the graph
    nx.draw(graph, with_labels=True, node_color='lightblue')
    plt.title("Problem Graph")
    plt.show()

    # Build required operators
    pauli_dict = build_all_paulis(graph.number_of_nodes())
    mixer_dict = build_all_mixers(graph)

    # Run ADAPT-QAOA with visualization
    results = run_adapt_qaoa(
        graph=graph,
        pauli_ops_dict=pauli_dict,
        gradient_ops_dict=mixer_dict,
        max_depth=5,
        beta_0=0.1,
        gamma_0=0.1,
        rel_gtol=1e-2
    )

    print("\nFinal Results:")
    print(f"Cut Approximation Ratios: {results['cut_approx_ratios']}")
    print(f"Best Mixers: {results['best_mixers']}")
  

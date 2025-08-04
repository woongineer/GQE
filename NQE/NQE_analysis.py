from analysis_utils import get_circuit_by_energy, get_circuit, get_data, make_random_circuit, population_data, \
    run_multiple_NQE_compare, plot_energy_errorbars

if __name__ == "__main__":
    circuit_filename = 'data_fix_sampling_SM_generated_circuit.json'
    data_filename = 'data_fix_sampling_SM_data_store.pkl'
    n_circuit = 3

    batch_size = 25
    N_layer = 1
    epoch = 8
    averaging_length = 4
    num_cpus = 4
    repeat = 6

    circuits = get_circuit(circuit_filename)
    data_x, data_y, _ = get_data(data_filename)
    original_x, original_y = population_data(train_len=400)

    good_circuits = get_circuit_by_energy(circuits, 'top', n_circuit=n_circuit)
    bad_circuits = get_circuit_by_energy(circuits, 'bottom', n_circuit=n_circuit)

    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'H', 'I']
    max_gate = 20
    num_qubits = 4

    rand_circuits = [make_random_circuit(gate_type, max_gate, num_qubits) for _ in range(n_circuit)]

    energy_list = run_multiple_NQE_compare(n_repeat=repeat,
                                           num_workers=num_cpus,
                                           data_x=data_x,
                                           data_y=data_y,
                                           N_layer=N_layer,
                                           batch_size=batch_size,
                                           epoch=epoch,
                                           good_circuits=good_circuits,
                                           bad_circuits=bad_circuits,
                                           rand_circuits=rand_circuits,
                                           ave_len=averaging_length)

    plot_energy_errorbars(energy_list, html_path="energy_errorbar.html", width=300)

"""Variational quantum circuit construction.

Kept deliberately torch-free: the circuit is built as a PennyLane QNode over
plain parameters, and the torch wrapping (qml.qnn.TorchLayer) happens in
hybrid.py. This lets the circuit be unit-tested without a torch install and
keeps the ablation axes (qubits, layers, embedding, noise) in one place.

Ansatz (matches the DCAS 2026 paper at n_qubits=3, n_layers=2,
embedding=amplitude): per layer, one qml.Rot(theta, phi, lam) on each qubit
followed by CNOTs in a ring topology; measurement is <Z_i> on every qubit.
Trainable parameter count = n_qubits * 3 * n_layers.
"""
from __future__ import annotations

import pennylane as qml
import numpy as np


def input_dim_for(embedding: str, n_qubits: int) -> int:
    """Encoder output dimension required by the chosen embedding."""
    if embedding == "amplitude":
        return 2 ** n_qubits
    if embedding == "angle":
        return n_qubits
    raise ValueError(f"Unknown embedding: {embedding}")


def _embed(inputs, embedding: str, n_qubits: int):
    if embedding == "amplitude":
        # Encoder ends in tanh -> values in [-1, 1]; AmplitudeEmbedding
        # l2-normalizes, matching the paper's normalization step.
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),
                               normalize=True, pad_with=0.0)
    elif embedding == "angle":
        # tanh output scaled to [-pi, pi] rotation angles.
        qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation="Y")
    else:
        raise ValueError(f"Unknown embedding: {embedding}")


def _entangle(n_qubits: int, topology: str):
    if n_qubits < 2:
        return
    if topology == "ring":
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    elif topology == "linear":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    else:
        raise ValueError(f"Unknown entangler topology: {topology}")


def _apply_channel_noise(noise_type: str, p: float, n_qubits: int):
    if noise_type == "none" or p <= 0.0:
        return
    for w in range(n_qubits):
        if noise_type == "depolarizing":
            qml.DepolarizingChannel(p, wires=w)
        elif noise_type == "amplitude_damping":
            qml.AmplitudeDamping(p, wires=w)
        else:
            raise ValueError(f"Unknown quantum noise type: {noise_type}")


def make_qnode(n_qubits: int,
               n_layers: int,
               embedding: str = "amplitude",
               entangler: str = "ring",
               noise_type: str = "none",
               noise_p: float = 0.0,
               device_name: str = "default.qubit",
               diff_method: str = "best",
               interface: str = "torch"):
    """Build the QNode. Returns (qnode, weight_shapes).

    weight_shapes is in the format expected by qml.qnn.TorchLayer:
    {"weights": (n_layers, n_qubits, 3)}.
    """
    if noise_type != "none" and device_name != "default.mixed":
        # Channel noise requires a mixed-state simulator.
        device_name = "default.mixed"

    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(inputs, weights):
        _embed(inputs, embedding, n_qubits)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.Rot(weights[layer, q, 0],
                        weights[layer, q, 1],
                        weights[layer, q, 2], wires=q)
            _entangle(n_qubits, entangler)
            _apply_channel_noise(noise_type, noise_p, n_qubits)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return circuit, weight_shapes


def n_quantum_params(n_qubits: int, n_layers: int) -> int:
    return n_qubits * 3 * n_layers

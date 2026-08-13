"""Model definitions: hybrid encoder-VQC-decoder, parameter-matched classical
NN, and non-neural baselines.

Architecture (DCAS 2026 defaults):
    Encoder: 12 -> hidden_dim -> q_in, LayerNorm + ReLU per layer,
             Dropout(p) after the first layer, Tanh on the output.
             q_in = 2**n_qubits (amplitude) or n_qubits (angle).
    VQC:     n_layers x [Rot per qubit + ring CNOT], <Z_i> outputs.
    Decoder: Linear(n_qubits -> 2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml

from .quantum import make_qnode, input_dim_for, n_quantum_params


def build_encoder(in_dim: int, hidden_dim: int, out_dim: int,
                  dropout: float, layer_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class HybridQNN(nn.Module):
    def __init__(self, cfg: dict, in_features: int = 12, n_classes: int = 2):
        super().__init__()
        qcfg = cfg["quantum"]
        ecfg = cfg["encoder"]
        self.n_qubits = qcfg["n_qubits"]
        q_in = input_dim_for(qcfg["embedding"], self.n_qubits)

        self.encoder = build_encoder(in_features, ecfg["hidden_dim"], q_in,
                                     ecfg["dropout"], ecfg["layer_norm"])

        qnode, weight_shapes = make_qnode(
            n_qubits=qcfg["n_qubits"],
            n_layers=qcfg["n_layers"],
            embedding=qcfg["embedding"],
            entangler=qcfg.get("entangler", "ring"),
            noise_type=qcfg.get("noise", {}).get("type", "none"),
            noise_p=qcfg.get("noise", {}).get("p", 0.0),
            device_name=qcfg.get("device", "default.qubit"),
            diff_method=qcfg.get("diff_method", "best"),
        )
        self.vqc = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.decoder = nn.Linear(self.n_qubits, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        q = self.vqc(z)
        return self.decoder(q)


class MatchedClassicalNN(nn.Module):
    """Same encoder/decoder; the VQC is replaced by a small classical block.

    The block maps q_in -> n_qubits with Tanh, mirroring the VQC's role as a
    bounded nonlinear bottleneck. Exact parameter parity with the VQC is not
    always achievable with dense layers (e.g. 18 quantum params vs 24+ for a
    Linear(8->3)); we therefore log exact counts per model and report both in
    the paper rather than claiming exact equality.
    """

    def __init__(self, cfg: dict, in_features: int = 12, n_classes: int = 2):
        super().__init__()
        qcfg = cfg["quantum"]
        ecfg = cfg["encoder"]
        n_qubits = qcfg["n_qubits"]
        q_in = input_dim_for(qcfg["embedding"], n_qubits)

        self.encoder = build_encoder(in_features, ecfg["hidden_dim"], q_in,
                                     ecfg["dropout"], ecfg["layer_norm"])
        self.bottleneck = nn.Sequential(
            nn.Linear(q_in, n_qubits, bias=False),
            nn.Tanh(),
        )
        self.decoder = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        return self.decoder(self.bottleneck(self.encoder(x)))


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum = sum(p.numel() for n, p in model.named_parameters()
                  if "vqc" in n and p.requires_grad)
    return {"total": total, "quantum": quantum, "classical": total - quantum}

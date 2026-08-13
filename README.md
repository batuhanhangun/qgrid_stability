# qgrid_stability

Code, per-run results, and analysis for the paper:

> B. Hangun, O. Altun, O. Eyecioglu, "Noise-Resilient Hybrid Quantum-Classical
> Learning for Smart Grid Stability: A Systematic Study of Architecture,
> Embedding, and Robustness," SN Computer Science (submitted, 2026). Extended
> version of the IEEE DCAS 2026 paper, doi: 10.1109/DCAS69364.2026.11544419.

The study evaluates a hybrid encoder-VQC-decoder architecture for smart grid
stability classification on the UCI Electrical Grid Stability dataset across
330 independent runs: a 30-seed main comparison against five classical
baselines, a reproduction of the conference protocol, an architecture and
embedding ablation, and a simulated quantum channel noise analysis.

## Repository contents

- `src/qgrid/`: the experimental framework (data, models, training,
  evaluation, statistics)
- `configs/`: base configuration and the four sweep definitions that generate
  all 330 runs
- `results/`: the complete per-run results (330 `result.json` files) from
  which every number, table, and figure in the paper derives
- `scripts/`: single-run entry point, sweep runner, aggregation, and the
  paper's figure and table generators

## Verify the paper's numbers without running anything

The shipped `results/` directory is the exact output of the experimental
campaign reported in the paper. To regenerate every table (with all
statistical tests) and every figure:

```bash
pip install -r requirements.txt
python scripts/make_tables.py     # prints tables + Wilcoxon/TOST/Holm stats
python scripts/make_figures.py    # writes the four paper figures as PDFs
```

Outputs are written to `paper_assets/`.

## Reproduce the experiments

```bash
pip install -r requirements.txt
python scripts/download_data.py   # fetches the UCI dataset (id 471)

# Single run (default = the paper's main hybrid configuration):
python scripts/run_experiment.py --config configs/base.yaml

# Any run group, sequentially (see table below for cost):
python scripts/run_sweep.py --sweep configs/sweeps/main_30seeds.yaml --execute
```

| Sweep | Paper section | Runs | Approx. CPU time |
|---|---|---|---|
| `subset_comparison.yaml` | Sec. 5.1 (reproduction) | 20 | minutes |
| `main_30seeds.yaml` | Sec. 5.2, 5.4 (main + input noise) | 180 | a few hours |
| `ablation_arch.yaml` | Sec. 5.3 (ablation) | 90 | 1 to 2 hours |
| `quantum_noise.yaml` | Sec. 5.5 (channel noise) | 40 | several hours (density-matrix simulation) |

Every run writes `results/<run_id>/result.json` containing the configuration,
parameter counts, training history, clean metrics, and the full input-noise
evaluation. Runs are independent and can be parallelized freely; any
scheduler or a simple process pool works, since each manifest line from
`run_sweep.py` is a self-contained command.

Aggregate and compare across seeds:

```bash
python scripts/aggregate_results.py --results results --metric accuracy
python scripts/aggregate_results.py --results results --compare hybrid_q3_l2_amplitude_n10000 classical_nn_q3_l2_amplitude_n10000
```

## Environment

The results in `results/` were produced with PennyLane 0.44, PyTorch 2.5.1,
scikit-learn, and XGBoost on CPU (containerized runs on NERSC Perlmutter CPU
nodes; the code has no HPC dependencies and runs identically on a laptop).
Exact per-run library behavior can differ slightly across platforms and
versions through RNG streams; the paper's statistical protocol (30 paired
seeds) is designed to make conclusions insensitive to this.

Seed semantics: the run seed controls the train/test split, weight
initialization, batch shuffling, and all noise draws. The 2,000-sample subset
used by the reproduction experiments is selected once with a fixed seed, so
all runs share identical subset membership.

## License and citation

MIT (see `LICENSE`). If you use this code or the results, please cite the
paper (see `CITATION.cff`).

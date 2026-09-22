# qgrid_stability

Code and per-run results for the paper *Noise-Resilient Hybrid Quantum-Classical
Learning for Smart Grid Stability: A Systematic Study of Architecture, Embedding,
and Robustness* (B. Hangun, O. Altun, O. Eyecioglu, SN Computer Science, 2026;
see `CITATION.cff`). `results/` holds the 1,430 runs the paper reports, one
`result.json` per run.

## Installation

Python 3.11.

```
pip install -r requirements.txt
```

CPU-only torch is sufficient (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## Reproduce the paper's tables and figures from the shipped results

```
python scripts/download_data.py      # writes data/grid_stability.csv (UCI dataset 471)
python scripts/make_tables.py        # prints all tables/statistics, writes paper_assets/tables/
python scripts/make_figures.py       # writes paper_assets/figures/
```

## Rerun experiments

Runs write to `runs/<run_id>/result.json` (`results/` is never touched).

Single run:

```
python scripts/run_experiment.py --config configs/base.yaml --set experiment.name=subset2k data.n_samples=2000 model.type=hybrid experiment.seed=0
```

Sweep (writes a manifest with one command per run, or executes them sequentially):

```
python scripts/run_sweep.py --sweep configs/sweeps/reproduction.yaml --manifest manifest.txt
python scripts/run_sweep.py --sweep configs/sweeps/reproduction.yaml --execute
```

| Sweep file | Paper section | Runs | Single-process CPU time |
|---|---|---|---|
| `reproduction.yaml` | Table 1 | 20 | ~10 min |
| `main.yaml` | Tables 2, 7, 8 (baseline); Figures 2, 5 | 180 | ~1.5 h |
| `channel_noise.yaml` | Table 8; Figure 7 | 40 | ~4 h |
| `ablation.yaml` | Tables 3, 4, 6; Figures 3, 4 | 540 | ~19 h |
| `ablation_classical.yaml` | Tables 4, 6 (classical control); Figure 3 | 540 | ~5 h |
| `mismatched_noise.yaml` | Table 9; Figure 6 | 70 | ~6 h |
| `parameter_control.yaml` | Table 1 (770/776-parameter rows) | 20 | ~10 min |
| `probes.yaml` | Table 5 | 20 | ~30 min |

Runs are independent: the lines of the manifest written by `run_sweep.py` can
be executed in parallel across processes or machines.

## Analyse your own reruns

```
python scripts/make_tables.py --results runs
python scripts/make_figures.py --results runs
```

## License

MIT, see `LICENSE`.

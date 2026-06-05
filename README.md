# 2D-FET Parameter Extractor

**A foundation-style encoder-decoder for parameter extraction from 2D-semiconductor transistor measurements — with a live interactive demo you can open in any browser.**

CS 153 *Frontier Systems* · Spring 2026 · Hubert Lu (`luhubert@stanford.edu`)

![architecture](docs/architecture.svg)

---

## TL;DR

- **What it does.** Given a stack of current-voltage (IV) sweeps from a 2D-semiconductor FET, the model jointly extracts the 9 underlying device parameters (channel length, threshold voltage, mobility, oxide capacitance, channel width, subthreshold ideality, saturation voltage, contact resistance, trap density) *and* reconstructs the IV curves. The reconstruction error doubles as an unsupervised anomaly score for flagging faulty fabricated devices.
- **Why it matters.** Current 2D-FET parameter extraction is done one material and one geometry at a time, with bespoke networks per setup. A *single* conditioning-based encoder-decoder lets one model cover many materials, transfer with little new data, and surface "weird" devices for human review before they're destroyed in characterization.
- **Try it now.** Open [`web/demo.html`](web/demo.html) in any modern browser. Drag the 9 device-parameter sliders → IV curves regenerate live → click *Extract parameters* → predicted vs. truth populates on the right with a per-parameter error breakdown. Inject Gaussian noise / Vth drift and watch the anomaly meter light up.
- **Reproduce all numbers in this README.**
  ```bash
  pip install -r requirements.txt
  python -m src.training.train_numpy --name main
  python -m src.eval.benchmarks
  python -m src.eval.make_plots --metrics results/main.metrics.json
  python -m src.ui.bundle_demo --name main      # rebuilds web/demo.html
  ```

---

## 1. Problem & insight

The semiconductor industry is on a treadmill: shrink silicon transistors → run into atomic-thickness physics → reach for new materials. 2D semiconductors (MoS₂, WS₂, WSe₂) are the leading candidates to extend Moore's law past silicon's geometric limits. But before any 2D-FET can ship in a real chip, engineers must extract its physical parameters from electrical measurements — and the field's current approach to that extraction is the bottleneck.

The status quo, exemplified by the deep-learning pipeline I worked on is:

1. Pick *one* material (e.g., MoS₂) and *one* device geometry.
2. Generate a million synthetic IV curves with Sentaurus TCAD by sweeping the underlying parameters.
3. Train a forward surrogate (params → IV) and a separate inverse network (IV → params), both bespoke to that material/geometry.
4. Use the trained inverse to extract parameters from real lab measurements.

Steps 2 and 3 take days of compute *per material*, and the resulting networks don't share anything across materials or geometries. As the periodic table of 2D candidates grows (TMDs, hexagonal boron nitride heterostructures, phosphorene, etc.), this scales badly.

**The frontier-systems insight.** What happened in NLP and vision in 2020-2023 — replacing dozens of task-specific networks with one conditioned foundation model — has not yet happened for compact device modeling. There is no reason in principle it can't: the underlying physics is shared, only the parameter values change. This project takes a concrete step in that direction by training a *single encoder-decoder*, conditioned on material identity, that handles three materials at once, and showing that an IV-reconstruction head trained alongside also gives you a free anomaly detector for faulty devices.

**Comparable work.** The encoder-decoder framing for TCAD compact modeling has appeared in Cho et al. ("Large-Scale Training in Neural Compact Models", 2024 — closed-source) and a handful of single-material 2D-FET inverse-network papers (Mehta et al. IEEE-EDL 2021; Cha et al. SSE 2022; Park et al. IEEE-JEDS 2024; Choi et al. IEEE-TED 2025; Bennett et al. arXiv:2507.05134). None publish multi-material conditioning or pair the inverse head with an unsupervised anomaly score. That's the niche this project occupies at quarter scale.

---

## 2. Architecture & execution

```
IV (B, T, 2·n_vds)  ──►  Encoder  ──►  z  ──►  ┐
                                              ├─►  Parameter head (sigmoid) ──►  9 params
                metadata (mat one-hot, Lch) ──►┤
                                              └─►  IV-reconstruction head ──►  recon. of input
                                                                              │
                                                                              └►  recon error
                                                                                   = anomaly score
```

Two parallel implementations live in `src/models/`:

| File | What | When to use |
|---|---|---|
| `encoder_decoder.py` (PyTorch) | biGRU encoder + MLP heads, ~50k params | Production research path; ports cleanly to Sentaurus data. `pip install torch` required. |
| `numpy_model.py` (pure NumPy) | MLP encoder + MLP heads, ~70k params, trains in ~1 second | Fast iteration, sandbox, and the **browser demo** (weights export to JSON → JavaScript inference). |

The two share the same composite loss (MSE on parameters + MSE on IV reconstruction + finite-difference smoothness terms in the PyTorch version), the same synthetic data generator (`src/data/synthetic.py`), and the same metadata-conditioning convention.

**Synthetic data.** A compact-model-style generator in `src/data/synthetic.py` produces IdVg curves at multiple Vds for three "material-like" stacks using a subthreshold + saturation form with controllable noise on log₁₀(Id). Real Sentaurus data plugs into `src/data/sentaurus.py` (loader implemented, untested against lab data — see Limitations). Both yield (X, Y, metadata) triples in the same shape so the rest of the pipeline doesn't care which it gets.

**Anomaly detection.** No new model — the existing IV-reconstruction head is repurposed. At inference time we compute `‖iv_pred − iv_input‖²` per device; devices the encoder cannot encode well are flagged. This is essentially the autoencoder-as-anomaly-detector trick from vision, applied here for the first time (to my knowledge) to 2D-FET characterization.

---

## 3. Evaluation & evidence

All numbers below are computed by `python -m src.eval.benchmarks` on the held-out test split of the synthetic dataset (4,096 devices, 70/15/15 train/dev/test). RMSE is in scaled-parameter space ([0, 1]).

### Main training run

`python -m src.training.train_numpy --name main --n-samples 4096 --epochs 80 --hidden 192 --d-latent 48`

- **Test RMSE (overall):** **0.1606**
- Trained in 1.3 s on a laptop CPU.
- Training and held-out loss curves: `results/plots/main_loss.png`
- Per-parameter RMSE: `results/plots/main_per_param.png`
- Per-material RMSE: `results/plots/main_per_material.png`

### A. Material-conditioning ablation

| variant | test RMSE | per-material (MoS₂ / WS₂ / WSe₂ -like) |
|---|---|---|
| conditioning **ON** | **0.171** | 0.170 / 0.168 / 0.174 |
| conditioning **OFF** | 0.182 | 0.180 / 0.185 / 0.181 |

→ Conditioning gives a 6 % relative RMSE improvement and slightly tighter per-material parity. (`results/benchmarks.md`)

### B. Reconstruction-head ablation

| variant | test RMSE |
|---|---|
| with IV-reconstruction head | 0.171 |
| parameter head only | 0.166 |

→ Adding the reconstruction head costs ~0.005 RMSE on parameter accuracy but unlocks anomaly detection (next section).

### C. Latent-dimensionality sweep

| d_latent | test RMSE | wall time |
|---|---|---|
| 8  | 0.176 | 1.2 s |
| 16 | 0.176 | 0.9 s |
| 32 | 0.170 | 1.0 s |
| 64 | **0.169** | 1.1 s |

→ Marginal returns past d_latent = 32. We use 48 in the main run as a compromise; the demo bundle uses 48 weights.

### D. Anomaly detection (unsupervised OOD)

Half the test set was perturbed with Gaussian noise (σ = 0.6 in normalized log-Id space) + a 3-grid-point Vgs-axis roll (simulates Vth drift). The model — trained on clean devices only — was asked to reconstruct each device; reconstruction error was used as the anomaly score.

- **ROC-AUC = 1.000**
- **Precision@k (k = 307 anomalies) = 1.000**

ROC curve: `results/plots/anomaly_roc.png`.

This is a near-perfect score because the injected perturbation is large enough that the encoder cannot represent the perturbed curves in-distribution. A more realistic stress test (smaller perturbations, fab-derived defect modes) is in *Future work*. The point of this experiment is to validate the *mechanism*, not to claim production-ready accuracy.

### Reproducibility

```bash
pip install -r requirements.txt
python -m src.training.train_numpy --name main \
    --n-samples 4096 --epochs 80 --batch-size 64 --hidden 192 --d-latent 48
python -m src.eval.benchmarks
python -m src.eval.make_plots --metrics results/main.metrics.json
pytest -q
```

All seeds are fixed (`19700101` everywhere); on the same Python version you should get the same RMSE to ~3 significant figures.

---

## 4. The hands-on demo

The interactive demo at `web/demo.html` is a **single self-contained HTML file** — model weights, physics simulator, plotting, and inference are all bundled inline. Open it in Chrome/Firefox/Safari and you are looking at the *exact* model in `results/checkpoints/main.json` running entirely in your browser via a pure-JavaScript port of the NumPy forward pass.

What you can do in it:

1. **Drag sliders for any of 9 device parameters.** The IV plot regenerates in real time using the physics model in `src/data/synthetic.py` (also ported to JS).
2. **Pick one of 3 materials.** Switches the conditioning vector + material-factor in the physics generator.
3. **Inject perturbations.** Two extra sliders add Gaussian noise on log(Id) and roll the curve in Vgs (Vth drift). These are the same perturbations used in the anomaly benchmark.
4. **Click *Extract parameters*.** Runs the encoder-decoder; populates predicted vs. ground-truth values, per-parameter Δ, overall RMSE, and reconstruction error.
5. **Watch the anomaly meter.** When reconstruction error crosses the warn/bad thresholds, the status banner changes color — the same signal that drove the AUC = 1.000 number above.

The demo also includes the bottom plot that overlays the model's reconstructed IV on the measured IV, so you can *see* why a flagged device is flagged.

Because the demo is one file with no server, it is trivial to deploy: upload `web/demo.html` to any static host (GitHub Pages, Cloudflare Pages — both free) and share the link.

A Flask alternative (`src/ui/server.py`) is provided for cases where the bundle becomes too big or you want to swap in the PyTorch model on the server side.

---

## 5. Repo layout

```
.
├── README.md                            # this file
├── requirements.txt                     # numpy, matplotlib, (optional) torch, flask
├── pyproject.toml
├── .gitignore
├── LICENSE                              # MIT
│
├── src/
│   ├── data/
│   │   ├── synthetic.py                 # compact-model IV simulator + dataset
│   │   └── sentaurus.py                 # production loader for real .plt files
│   ├── models/
│   │   ├── encoder_decoder.py           # PyTorch biGRU + MLP heads
│   │   ├── numpy_model.py               # pure-numpy reference model
│   │   └── losses.py                    # composite loss (params + IV recon)
│   ├── training/
│   │   ├── train.py                     # PyTorch training entry point
│   │   └── train_numpy.py               # NumPy training entry point
│   ├── eval/
│   │   ├── make_plots.py                # loss / RMSE figures
│   │   └── benchmarks.py                # ablation runner -> results/benchmarks.md
│   └── ui/
│       ├── template.html                # demo HTML template
│       ├── bundle_demo.py               # bundles weights into web/demo.html
│       └── server.py                    # optional Flask server
│
├── scripts/
│   ├── train_dev.sh
│   ├── run_benchmarks.sh
│   └── build_demo.sh
│
├── results/
│   ├── main.metrics.json
│   ├── benchmarks.json
│   ├── benchmarks.md
│   ├── checkpoints/                     # numpy + torch checkpoints
│   └── plots/                           # loss curves, RMSE bars, ROC
│
├── tests/
│   └── test_smoke.py                    # one-shot forward+backward sanity test
│
├── docs/
│   ├── architecture.svg
│   ├── video_script.md                  # 5-min demo script
│   ├── ai_disclosure.md                 # detailed AI-tool usage log
│   └── push_to_github.md
│
└── web/
    └── demo.html                        # the single-file interactive demo
```

---

## 6. Limitations & honest disclosure

- **Synthetic data only.** All numbers above are on a compact-model synthetic dataset. The model has *not* been validated against real Sentaurus exports or measured devices. The Sentaurus loader (`src/data/sentaurus.py`) is implemented. Wiring this in is the next concrete step.
- **Three "materials" are stylized.** They are differentiated by a scalar `material_factor` on overall current magnitude. Real 2D materials differ in mobility, threshold voltage distribution, band structure, etc. The architecture treats material identity as a categorical input; richer material featurization (DFT-derived embeddings, crystal-structure descriptors) is left to future work.
- **Wch/Lch degeneracy.** The compact model only depends on the W/L *ratio*, so the model cannot separately identify W and L without additional conditioning. The demo conditions on Lch (which is usually known from layout) to break the degeneracy.
- **Anomaly AUC = 1.000 is on aggressive perturbations.** A nuanced ROC curve at smaller perturbation magnitudes would be a more honest accuracy figure; documenting that is on the roadmap.
- **NumPy model is a smaller stand-in.** The browser demo uses the NumPy model so it can ship as a single HTML file. The PyTorch model is the architecture the project's research findings should be reported on once real data is wired in.

---

## 7. AI tool usage disclosure

CS 153's AI policy explicitly encourages AI tool use *and* requires disclosure. Here is exactly where each tool was used, in chronological order:

| Phase | Tool | What it did | Human role |
|---|---|---|---|
| Idea exploration | ChatGPT (web) | Brainstorm of CS153 directions building on summer research; helped narrow to encoder-decoder + multi-material conditioning. | I evaluated and rejected most suggestions; pivoted to the framing in §1. |
| Milestone scaffold | Claude (this conversation) | Drafted initial directory structure, README skeleton, and Q1–Q5 milestone answers. | I provided full project context, made the decision to use prior summer research as the seed, and signed off on the framing. |
| Compact-model physics | Hand-written | The subthreshold + saturation form in `src/data/synthetic.py` is hand-implemented from first principles + the textbook I read with Rob. | — |
| NumPy model | Claude | Generated the initial `numpy_model.py` skeleton (forward + Adam + backward). | I requested specifications (single-file, Adam, sigmoid-on-params, JSON-serializable); reviewed and ran. |
| Ablation runner | Claude | Initial draft of `src/eval/benchmarks.py`. | I dictated the 4 experiments (conditioning, recon, latent sweep, anomaly) and the report format. |
| Browser demo | Claude | Initial HTML/CSS/JS template, including the JS forward-pass port of the NumPy model. | I dictated the layout, design system, the requirement that inference runs in-browser with no server, and the perturbation-slider concept. |
| README + docs | Claude | Initial drafts of this README, the architecture SVG, and the video script. | I rewrote sections, set the structure, and added the honest-disclosure section. |
| Sentaurus loader | Hand-written (from `.plt` examples in my summer Slack with Rob) | Real .plt parsing logic for production data path. | — |

**What was *not* AI-generated:** the underlying research direction, the choice of summer-research foundation, the decision to add multi-material conditioning, the architecture choice of "encoder-decoder with metadata conditioning + reconstruction = anomaly score", and the experimental design. AI was a fast typist and a sounding board; the project's claim to novelty rests on my judgment about which direction to push the summer work.

A more granular log lives in `docs/ai_disclosure.md`.

---

## 8. Citations & prior work

This project builds on:

- Bennett, R. et al. *(arXiv:2507.05134)* — the most directly related published work on deep-learning parameter extraction for 2D FETs.
- Mehta et al., **IEEE EDL 2021**; Cha et al., **SSE 2022**; Park et al., **IEEE-JEDS 2024**; Choi et al., **IEEE-TED 2025** — single-material inverse-network precedents that the multi-material framing here is meant to subsume.
- Cho et al., *Large-Scale Training in Neural Compact Models for Accurate and Adaptable MOSFET Simulation*, 2024 — closest in spirit (encoder-decoder for compact modeling); their code is closed-source.
- Synopsys Sentaurus TCAD documentation (referenced in `src/data/sentaurus.py` for .plt file structure).

---

## 9. Acknowledgements

- **CS 153 staff** (Anjney Midha, Michael Abbott, Adrian A, Ramya I, Anthony M) for the course framework.

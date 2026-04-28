# Model Card: heteroscedastic_unet

## Intended use

DL-AMR refinement indicator for two-dimensional bluff-body wake flows in the
laminar regime. The model predicts per-grid-point aleatoric uncertainty
$\hat{\sigma}$ used to flag mesh cells for refinement at runtime in OpenFOAM.

## Training data

- **Geometry**: circular cylinder
- **Reynolds number**: $Re = 100, 150$
- **Snapshots**: ~400 (200 per $Re$, $\Delta t_{\mathrm{write}} = 1$, $t \in [100, 300]$)
- **Grid**: $64 \times 224$ uniform Cartesian, wake-cropped to
  $x/D \in [2, 39]$, $y/D \in [-5, 5]$
- **Channels (input)**: $(u^\ast, v^\ast, p^\ast)$, normalised to zero mean
  and unit variance per channel
- **Target**: temporal residual $\Delta\mathbf{q}_t = \mathbf{q}_{t+1} - \mathbf{q}_t$
- **Train/val/test split**: 80/10/10

## Architecture

- 4-level U-Net encoder–decoder
- Channel progression: 32 → 64 → 128 → 256, bottleneck 512
- Two output heads:
  - Mean head: predicted temporal residual ($3 \times H \times W$)
  - Log-variance head: scalar log-variance map ($1 \times H \times W$),
    clamped to $[-12, 6]$
- ~7.8 M trainable parameters

## Training

- Loss: heteroscedastic Gaussian negative log-likelihood (NLL)
- Optimiser: Adam, lr = $3 \times 10^{-4}$, batch 32, 100 epochs
- Best checkpoint selected by lowest validation NLL (epoch 54)

## Performance (held-out test set)

- Spearman rank correlation $\rho = 0.906$ between $\hat{\sigma}$ and $|\Delta\mathbf{q}|$
- Area under ROC curve (top 5% targets): AUC = 0.952
- Calibration: near-diagonal in observed-vs-predicted RMSE plot

## Refinement use

- Threshold $\sigma_{\mathrm{thr}} = 0.6$ (selected from threshold sensitivity
  analysis; captures ~1.3% of grid points with refined/unrefined contrast
  ratio above $9 \times$).

## Cross-geometry transfer (without retraining)

| Geometry         | AUC ($q_{95}$) |
|------------------|----------------|
| Circular (train) | 0.952          |
| Square (unseen)  | 0.787          |
| Diamond (unseen) | 0.774          |

Wake-field $L_2$ error of $\overline{U}_x$ is reduced over the coarse and
gradient-based AMR baselines for all three geometries.

## Limitations

- 2D laminar regime only; not validated for 3D or turbulent flows
- Trained on a single body shape (circular); transfer to non-circular
  geometries shows partial AUC degradation
- Wake-cropped input domain ($x/D \in [2, 39]$); behaviour outside this
  region is undefined
- Refinement decision is binary; the model does not provide finer
  multi-level refinement guidance

## Software requirements

- PyTorch ≥ 2.0 (export and inference)
- TorchScript-compatible LibTorch C++ runtime for OpenFOAM integration
- OpenFOAM v2312 with the `amrPimpleFoam` solver in this repository

## License

The model weights are released under CC-BY 4.0. See [`../LICENSE-DATA`](../LICENSE-DATA).

## Citation

If you use this model, please cite the accompanying paper:
Kang et al., *Journal of Computational Physics*, 2026. DOI: `10.xxxx/yyyy`.

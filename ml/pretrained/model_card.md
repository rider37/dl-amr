# Model Card: heteroscedastic_unet

## Intended use

DL-AMR refinement indicator for two-dimensional bluff-body wake flows in the
laminar regime. The model predicts per-grid-point aleatoric uncertainty
$\hat{\sigma}$ used to flag mesh cells for refinement at runtime in OpenFOAM.

## Training data

- **Geometry**: circular cylinder
- **Reynolds number**: $Re = 100, 150$
- **Snapshots**: ~400 (200 per $Re$, $\Delta t_{\mathrm{write}} = 1$, $t \in [100, 300]$)
- **Grid**: $432 \times 160$ uniform Cartesian covering
  $x/D \in [-2, 25]$, $y/D \in [-5, 5]$
- **Channels (input)**: $(u^\ast, v^\ast)$, normalised to zero mean and unit
  variance per channel. Pressure is excluded because the remapping transients of
  dynamic refinement contaminate it.
- **Target**: mesh-induced residual
  $\Delta\mathbf{q} = \mathbf{q}^{\mathrm{fine}} - \mathbf{q}^{\mathrm{coarse}}$,
  the pointwise difference between the solution on a refined reference mesh
  (135,840 cells) and the solution on the coarse mesh that the policy refines.
  Snapshots are paired by instantaneous shedding phase, not by wall-clock time.
- **Train/val/test split**: 80/10/10 over **contiguous chronological blocks**
  (not random). The test block $t \in [375, 400]$ is separated from the last
  training sample by 25.2 convective time units, about four shedding periods.
- **Data volume**: 1,251 pairs per Reynolds number ($Re = 100, 150$), 2,502 in total.

## Architecture

- 4-level U-Net encoder-decoder
- Channel progression: 32 -> 64 -> 128 -> 256, bottleneck 512
- Two output heads:
  - Mean head: predicted mesh-induced residual ($2 \times H \times W$)
  - Log-variance head: scalar log-variance map ($1 \times H \times W$),
    clamped to $[-12, 6]$
- ~7.8 M trainable parameters

## Training

- Loss: heteroscedastic Gaussian negative log-likelihood (NLL)
- Optimiser: Adam, lr = $3 \times 10^{-4}$
- Best checkpoint selected by lowest validation NLL

## Performance (held-out blocked test set)

- Spearman rank correlation $\rho = 0.809$ between $\hat{\sigma}$ and
  $|\Delta\mathbf{q}|$, and $\rho = 0.845$ against the mean-head residual
  magnitude $\lVert\widehat{\Delta\mathbf{q}} - \Delta\mathbf{q}\rVert$,
  which is used to assess the fitted error scale
- Area under ROC curve: AUC = 0.859 ($q_{90}$), 0.848 ($q_{95}$),
  0.845 ($q_{98}$), 0.852 ($q_{99}$)
- Calibration: the observed RMSE is close to proportional to $\hat{\sigma}$
  (log-log slope 1.10) but offset by a factor of 3.3 (median over bins). The
  indicator is therefore used as a ranking score, not as an absolute error estimate.

## Refinement use

Cells are sorted by decreasing $\hat{\sigma}$ and marked in order until a
prescribed cell-count budget is reached (Eq. (11) of the paper). No magnitude
threshold is calibrated: the budget is set per geometry so that the realised
mean cell count stays at or below the 49,884 cells of the static wake mesh.

## Cross-geometry transfer (without retraining)

| Geometry         | AUC ($q_{95}$) | near wake | Spearman $\rho$ |
|------------------|----------------|-----------|------------------|
| Circular (train) | 0.848          | 0.783     | 0.809            |
| Square (unseen)  | 0.800          | 0.859     | 0.805            |
| Diamond (unseen) | 0.916          | 0.752     | 0.728            |

At a common cell budget the closed-loop policy gives the lowest wake-field
$L_2$ error of $\overline{U}_x$ of all variants on the circular and square
bodies and is within about 1% of the best classical indicator on the diamond
(Table 1 of the paper). The score ablation (Section 4.6) exchanges the
variance head for the mean head of the same network: the variance score is
more accurate in $\overline{U}_x$ and $\overline{\omega}_z$ on the square
and diamond bodies, the mean head in several fluctuation and frequency
metrics, so the advantage is metric-dependent.

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

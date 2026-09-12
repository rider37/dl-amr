# Pretrained Models

This directory holds **download instructions** only. Actual model weights
are distributed via GitHub Release and Zenodo for archival stability.

## Files (after download)

| File                                  | Type          | Approx. Size | Description                                                  |
|---------------------------------------|---------------|-------------:|--------------------------------------------------------------|
| `heteroscedastic_unet.pt`             | TorchScript   | 31 MB        | DL-AMR refinement indicator used in the paper. Loaded by `amrPimpleFoam` at runtime; wraps the network with the target-normalisation buffers, so it returns the residual in physical units. |
| `heteroscedastic_unet_state_dict.pt`  | `state_dict`  | 31 MB        | Same weights, for offline evaluation with `ml.src.models.hetero_model.HeteroDeltaFullRes`. |
| `train_config.json`                   | JSON          | < 1 KB       | Training configuration of the run that produced these weights. |
| `norm_stats.json`, `target_norm.json` | JSON          | < 1 KB       | Per-channel input and target normalisation statistics.        |
| `blocked_split_ids.json`              | JSON          | 14 KB        | Snapshot indices of the contiguous chronological train/val/test blocks. |
| `vort_grid.ts`, `make_vort_wrapper.py` | TorchScript / Python | 4 KB | Vorticity wrapper of the score ablation (Section 4.6): returns `log(omega_z^2 + 1e-12)` on the sampling grid; contains no trained weights. |

The SHA-256 checksums of the archives are distributed with the deposit as `SHA256SUMS.txt`.

## Download

### Recommended: Zenodo (DOI-cited, long-term archival)

The pretrained model is bundled in the same Zenodo deposit as the code,
under the concept DOI
[10.5281/zenodo.19870610](https://doi.org/10.5281/zenodo.19870610),
which always resolves to the latest archived version.

`scripts/download_models.sh` resolves the concept DOI to the latest
version record at runtime (via the Zenodo REST API), so it does not need
to be edited when a new release is published. To pin a specific version,
override `ZENODO_RECORD`:

```bash
# Latest (default)
make download-models

# Pinned to the v1.1.0 version record
ZENODO_RECORD=22724267 make download-models
```

Manual download (also points at the latest version via concept DOI):

```bash
# Resolve the latest version record ID, then fetch the binary.
LATEST=$(curl -sL https://zenodo.org/api/records/19870610 \
    | grep -oE '"id"[[:space:]]*:[[:space:]]*[0-9]+' | head -1 | grep -oE '[0-9]+')
wget "https://zenodo.org/records/${LATEST}/files/pretrained_models.tar.gz"
sha256sum pretrained_models.tar.gz   # verify against above (after upload)
tar -xzf pretrained_models.tar.gz -C ml/pretrained/
```

### Alternative: GitHub Release (mirror)

```bash
gh release download --repo rider37/dl-amr \
    --pattern pretrained_models.tar.gz   # latest release
tar -xzf pretrained_models.tar.gz -C ml/pretrained/
```

## Usage (Python)

```python
import torch
model = torch.jit.load('ml/pretrained/heteroscedastic_unet.pt')
model.eval()
# Input: (B, 2, H, W) tensor of (u*, v*) channels on the 432 x 160 grid,
# normalised with norm_stats.json
mean_pred, logvar_pred = model(x)
sigma = torch.sqrt(torch.exp(logvar_pred))   # uncertainty map
```

## Usage (OpenFOAM `amrPimpleFoam`)

The case templates under `cases/<geometry>/dl_amr/` expect the model at
`constant/model.ts`. Symlink or copy after download:

```bash
ln -s $(pwd)/ml/pretrained/heteroscedastic_unet.pt \
      cases/circular_Re200/dl_amr/constant/model.ts
```

## Model card

See [`model_card.md`](model_card.md) for training data, architecture,
performance, and limitations.

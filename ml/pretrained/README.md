# Pretrained Models

This directory holds **download instructions** only. Actual model weights
are distributed via GitHub Release and Zenodo for archival stability.

## Files (after download)

| File                          | Type        | Approx. Size | SHA256        | Description                                       |
|-------------------------------|-------------|-------------:|---------------|---------------------------------------------------|
| `heteroscedastic_unet.pt`     | TorchScript | 31 MB        | _to be filled after final artifact upload_ | DL-AMR refinement indicator used in the paper     |

## Download

### Recommended: Zenodo (DOI-cited, long-term archival)

The pretrained model is bundled in the same Zenodo deposit as the code:

- Concept DOI (always resolves to latest version):
  [10.5281/zenodo.19870610](https://doi.org/10.5281/zenodo.19870610)
- Pinned version `v1.0.2`:
  [10.5281/zenodo.19873110](https://doi.org/10.5281/zenodo.19873110)

```bash
wget https://zenodo.org/records/19873110/files/pretrained_models.tar.gz
sha256sum pretrained_models.tar.gz   # verify against above
tar -xzf pretrained_models.tar.gz -C ml/pretrained/
```

### Alternative: GitHub Release (mirror)

```bash
gh release download v1.0.2 --repo rider37/dl-amr \
    --pattern pretrained_models.tar.gz
tar -xzf pretrained_models.tar.gz -C ml/pretrained/
```

## Usage (Python)

```python
import torch
model = torch.jit.load('ml/pretrained/heteroscedastic_unet.pt')
model.eval()
# Input: (B, 3, H, W) tensor of (u*, v*, p*) channels, normalised per train stats
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

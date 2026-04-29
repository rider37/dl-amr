# Pretrained Models

This directory holds **download instructions** only. Actual model weights
are distributed via GitHub Release and Zenodo for archival stability.

## Files (after download)

| File                          | Type        | Approx. Size | SHA256        | Description                                       |
|-------------------------------|-------------|-------------:|---------------|---------------------------------------------------|
| `heteroscedastic_unet.pt`     | TorchScript | 31 MB        | _to be filled after final artifact upload_ | DL-AMR refinement indicator used in the paper     |

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

# Pinned to a specific Zenodo version record
ZENODO_RECORD=<record-id> make download-models
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

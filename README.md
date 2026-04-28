# JAGUAR

**JAGUAR** is **J**oint **AGN-Galaxy Unified Analysis & Reconstruction**.

The package provides JAX/NumPyro tools for joint unresolved AGN, PSF,
host-galaxy image fitting, and grahspj SED coupling. The initial workflow is
modeled on the galight HSC QSO notebook: load a science image, noise image, WCS
target position, and explicit PSF; fit a point-source AGN plus Sersic host with
band fluxes supplied by the same grahspj model used for the SED likelihood.

## Status

This is an initial scaffold with a working pure-JAX image likelihood and unified
grahspj coupling. The image renderer is kept small and deterministic so
flux-normalization and masking invariants can be tested without a heavy runtime
environment.

## Local Development

JAGUAR requires `grahspj` for normal fitting. Install the requirements or install
both local packages editable:

```bash
python -m pip install -r requirements.txt
python -m pip install -e /Users/colinburke/research/grahspj
python -m pip install -e /Users/colinburke/research/jaguar
```

## Quick Sketch

```python
from jaguar import (
    ImageFitConfig,
    JointFitConfig,
    build_grahspj_config_from_image_bands,
    fit,
)

grahspj_cfg = build_grahspj_config_from_image_bands(
    image_bands,
    object_id="example",
    redshift=0.5,
    dsps_ssp_fn="/Users/colinburke/research/jaxqsofit/tempdata.h5",
)
joint = JointFitConfig(
    image_bands=image_bands,
    image=ImageFitConfig(),
    grahspj_config=grahspj_cfg,
)

result = fit(joint, fit_method="map_only")
```

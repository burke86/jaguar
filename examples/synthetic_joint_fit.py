from __future__ import annotations

import numpy as np

from jaguar import ComponentFluxes, ImageBandData, ImageFitConfig, JointFitConfig, fit
from jaguar.model import render_joint_model


def main() -> None:
    shape = (41, 41)
    yy, xx = np.mgrid[:9, :9]
    psf = np.exp(-0.5 * ((xx - 4) ** 2 + (yy - 4) ** 2) / 1.2**2)
    noise = np.ones(shape) * 2.0
    blank = np.zeros(shape)
    fluxes = {"hsc_i": ComponentFluxes(agn=500.0, host=2000.0)}
    band = ImageBandData(blank, noise, psf, "hsc_i", pixel_scale=0.168)
    cfg = JointFitConfig([band], ImageFitConfig(fit_background=False), fixed_component_fluxes=fluxes)
    params = {"host_reff_arcsec": 0.4, "host_n_sersic": 2.0, "host_e1": 0.1, "host_e2": 0.0, "center_x_pix": 0.0, "center_y_pix": 0.0}
    image = np.asarray(render_joint_model(cfg, params)["hsc_i"]["total"])
    band = ImageBandData(image, noise, psf, "hsc_i", pixel_scale=0.168)
    cfg = JointFitConfig([band], ImageFitConfig(fit_background=False), fixed_component_fluxes=fluxes)
    result = fit(cfg, fit_method="map_only", steps=100)
    print(result.summary())


if __name__ == "__main__":
    main()


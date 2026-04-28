from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def normalize_image(image: jnp.ndarray) -> jnp.ndarray:
    """Return an image normalized to unit total flux."""

    total = jnp.sum(jnp.asarray(image, dtype=jnp.float64))
    return jnp.asarray(image, dtype=jnp.float64) / jnp.maximum(total, 1.0e-30)


def convolve_fft_same(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """FFT-convolve an image with a centered kernel and return the image-sized result."""

    image = jnp.asarray(image, dtype=jnp.float64)
    kernel = normalize_image(jnp.asarray(kernel, dtype=jnp.float64))
    ny, nx = image.shape
    ky, kx = kernel.shape
    full_shape = (ny + ky - 1, nx + kx - 1)
    padded_image = jnp.pad(image, ((0, ky - 1), (0, kx - 1)))
    padded_kernel = jnp.pad(kernel, ((0, ny - 1), (0, nx - 1)))
    convolved = jnp.fft.irfftn(
        jnp.fft.rfftn(padded_image, s=full_shape) * jnp.fft.rfftn(padded_kernel, s=full_shape),
        s=full_shape,
    )
    start_y = (ky - 1) // 2
    start_x = (kx - 1) // 2
    same = lax.dynamic_slice(convolved, (start_y, start_x), (ny, nx))
    return normalize_image(jnp.clip(same, 0.0, jnp.inf))


def pixel_coordinates(shape: tuple[int, int], pixel_scale: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Centered pixel coordinates in arcsec."""

    ny, nx = shape
    y = (jnp.arange(ny, dtype=jnp.float64) - (ny - 1) / 2.0) * pixel_scale
    x = (jnp.arange(nx, dtype=jnp.float64) - (nx - 1) / 2.0) * pixel_scale
    return jnp.meshgrid(x, y)


def ellipticity_to_q_phi(e1: jnp.ndarray, e2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert lenstronomy-style ellipticity components to axis ratio and PA."""

    ellip = jnp.clip(jnp.sqrt(e1 * e1 + e2 * e2 + 1.0e-12), 0.0, 0.85)
    q = (1.0 - ellip) / (1.0 + ellip)
    phi = 0.5 * jnp.arctan2(e2, e1 + 1.0e-6)
    return q, phi


def sersic_ellipse_unit_flux(
    shape: tuple[int, int],
    pixel_scale: float,
    reff_arcsec: jnp.ndarray,
    n_sersic: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    center_x_pix: jnp.ndarray = 0.0,
    center_y_pix: jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """Render a unit-flux elliptical Sersic image on the native pixel grid."""

    xx, yy = pixel_coordinates(shape, pixel_scale)
    x0 = center_x_pix * pixel_scale
    y0 = center_y_pix * pixel_scale
    x = xx - x0
    y = yy - y0
    q, phi = ellipticity_to_q_phi(e1, e2)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    x_rot = x * cos_phi + y * sin_phi
    y_rot = -x * sin_phi + y * cos_phi
    radius = jnp.sqrt(x_rot * x_rot + (y_rot / jnp.maximum(q, 1.0e-3)) ** 2 + 1.0e-12)
    n = jnp.clip(n_sersic, 0.3, 8.0)
    reff = jnp.maximum(reff_arcsec, 1.0e-3)
    b_n = 2.0 * n - 1.0 / 3.0 + 0.009876 / n
    image = jnp.exp(-b_n * ((radius / reff) ** (1.0 / n) - 1.0))
    return normalize_image(jnp.clip(image, 0.0, jnp.inf))


def shift_image_bilinear(image: jnp.ndarray, dx_pix: jnp.ndarray, dy_pix: jnp.ndarray) -> jnp.ndarray:
    """Shift an image by sub-pixel offsets using bilinear interpolation."""

    image = jnp.asarray(image, dtype=jnp.float64)
    ny, nx = image.shape
    yy, xx = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
    src_x = xx - dx_pix
    src_y = yy - dy_pix
    x0 = jnp.floor(src_x).astype(jnp.int32)
    y0 = jnp.floor(src_y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = src_x - x0
    wy = src_y - y0

    def sample(ix: jnp.ndarray, iy: jnp.ndarray) -> jnp.ndarray:
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        values = image[jnp.clip(iy, 0, ny - 1), jnp.clip(ix, 0, nx - 1)]
        return jnp.where(valid, values, 0.0)

    shifted = (
        sample(x0, y0) * (1.0 - wx) * (1.0 - wy)
        + sample(x1, y0) * wx * (1.0 - wy)
        + sample(x0, y1) * (1.0 - wx) * wy
        + sample(x1, y1) * wx * wy
    )
    return normalize_image(jnp.clip(shifted, 0.0, jnp.inf))


def psf_unit_flux(psf: jnp.ndarray, shape: tuple[int, int], dx_pix: jnp.ndarray = 0.0, dy_pix: jnp.ndarray = 0.0) -> jnp.ndarray:
    """Center a PSF image in a science stamp and normalize it to unit flux."""

    psf = normalize_image(psf)
    ny, nx = shape
    py, px = psf.shape
    canvas = jnp.zeros(shape, dtype=jnp.float64)
    y0 = (ny - py) // 2
    x0 = (nx - px) // 2
    canvas = canvas.at[y0 : y0 + py, x0 : x0 + px].set(psf)
    return shift_image_bilinear(canvas, dx_pix, dy_pix)

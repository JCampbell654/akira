import jax.numpy as jnp
import numpy as np
import pytest

from akira.imaging import ImagingObservation, PSF, convolve_psf, gaussian_psf


def test_gaussian_psf_is_odd_nonnegative_and_normalized() -> None:
    psf = gaussian_psf(sigma_pixels=1.2)
    assert psf.kernel.shape[0] % 2 == 1
    assert psf.kernel.shape[0] == psf.kernel.shape[1]
    assert np.all(np.asarray(psf.kernel) >= 0)
    np.testing.assert_allclose(psf.kernel.sum(), 1.0, rtol=1e-14, atol=1e-14)


def test_delta_psf_leaves_image_unchanged() -> None:
    image = jnp.arange(30, dtype=jnp.float64).reshape(5, 6)
    np.testing.assert_array_equal(convolve_psf(image, PSF.delta().kernel), image)


def test_psf_convolution_preserves_central_point_flux() -> None:
    image = np.zeros((21, 21))
    image[10, 10] = 3.5
    convolved = convolve_psf(image, gaussian_psf(1.0, size=9).kernel)
    np.testing.assert_allclose(convolved.sum(), 3.5, rtol=1e-14, atol=1e-14)


def test_psf_convolution_does_not_wrap_image_edges() -> None:
    image = np.zeros((15, 15))
    image[0, 0] = 1.0
    convolved = np.asarray(convolve_psf(image, gaussian_psf(1.0, size=7).kernel))
    assert convolved[0, 0] > 0
    np.testing.assert_allclose(convolved[-1, -1], 0.0, atol=1e-15)


@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((2, 3)) / 6,
        np.ones((3, 3)),
        np.array([[0.0, -1.0, 2.0]]),
    ],
)
def test_invalid_psfs_are_rejected(kernel: np.ndarray) -> None:
    with pytest.raises(ValueError):
        PSF(kernel)


def test_observation_validates_and_materializes_default_mask() -> None:
    observation = ImagingObservation(
        data=np.ones((3, 4)),
        noise_map=np.full((3, 4), 0.1),
        psf=PSF.delta(),
        pixel_scale=0.05,
    )
    assert observation.mask.shape == observation.data.shape
    assert not np.asarray(observation.mask).any()
    assert observation.data.dtype == jnp.float64


def test_observation_rejects_nonpositive_noise() -> None:
    with pytest.raises(ValueError):
        ImagingObservation(
            data=np.ones((2, 2)),
            noise_map=np.zeros((2, 2)),
            psf=PSF.delta(),
            pixel_scale=0.1,
        )


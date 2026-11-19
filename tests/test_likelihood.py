import jax
import jax.numpy as jnp
import numpy as np

from akira.likelihood import chi_squared, gaussian_log_likelihood


def numpy_log_likelihood(data, model, noise, mask):
    term = ((data - model) / noise) ** 2 + np.log(2 * np.pi * noise**2)
    return -0.5 * term[~mask].sum()


def test_log_likelihood_matches_independent_numpy_equation() -> None:
    data = np.array([[1.0, 1.3], [0.7, -0.2]])
    model = np.array([[0.9, 1.4], [0.4, 0.1]])
    noise = np.array([[0.1, 0.2], [0.3, 0.4]])
    mask = np.array([[False, True], [False, False]])

    expected = numpy_log_likelihood(data, model, noise, mask)
    actual = gaussian_log_likelihood(data, model, noise, mask)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_identical_model_has_zero_chi_squared() -> None:
    data = jnp.arange(12, dtype=jnp.float64).reshape(3, 4)
    noise = jnp.ones_like(data) * 0.2
    mask = jnp.zeros_like(data, dtype=bool)
    assert float(chi_squared(data, data, noise, mask)) == 0.0


def test_masked_pixel_has_no_effect() -> None:
    data = jnp.array([0.0, 1e12])
    model = jnp.zeros(2)
    noise = jnp.ones(2)
    mask = jnp.array([False, True])
    masked = gaussian_log_likelihood(data, model, noise, mask)
    reference = gaussian_log_likelihood(
        jnp.array([0.0]), jnp.array([0.0]), jnp.array([1.0]), jnp.array([False])
    )
    np.testing.assert_allclose(masked, reference)


def test_likelihood_gradient_matches_finite_difference() -> None:
    data = jnp.array([0.2, 0.7, -0.1])
    noise = jnp.array([0.1, 0.2, 0.3])
    mask = jnp.array([False, False, False])

    def likelihood(parameter):
        model = jnp.array([parameter, 2 * parameter, -parameter])
        return gaussian_log_likelihood(data, model, noise, mask)

    point = 0.25
    autodiff = float(jax.grad(likelihood)(point))
    step = 1e-6
    finite = float((likelihood(point + step) - likelihood(point - step)) / (2 * step))
    np.testing.assert_allclose(autodiff, finite, rtol=1e-8, atol=1e-8)


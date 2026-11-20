import numpy as np
import pytest

from akira import (
    CartesianGrid,
    ImagingObservation,
    LensPlane,
    PSF,
    SourceGrid,
    gaussian_source_log_evidence,
    lensing_operator,
    optimize_regularization,
    regularization_matrix,
)


class ZeroDeflection:
    def deflection(self, coordinates):
        return coordinates * 0.0


def make_identity_problem():
    grid = CartesianGrid((6, 6), 0.1)
    source_grid = SourceGrid((6, 6), 0.1)
    coordinates = source_grid.coordinates()
    x, y = coordinates[..., 0], coordinates[..., 1]
    source = np.exp(-0.5 * (x**2 / 0.11**2 + y**2 / 0.08**2))
    rng = np.random.default_rng(11)
    noise = 0.08
    observation = ImagingObservation(
        data=source + rng.normal(0.0, noise, source.shape),
        noise_map=np.full(source.shape, noise),
        psf=PSF.delta(),
        pixel_scale=grid.pixel_scale,
    )
    operator = lensing_operator(grid, source_grid, LensPlane((ZeroDeflection(),)))
    return source_grid, observation, operator


@pytest.mark.parametrize("kind", ["zeroth", "gradient", "curvature"])
def test_suyu_regularization_is_positive_definite(kind):
    matrix = np.asarray(regularization_matrix(SourceGrid((5, 6), 0.1), kind))
    assert np.linalg.eigvalsh(matrix).min() > 0.0


@pytest.mark.parametrize("kind", ["zeroth", "gradient", "curvature"])
def test_log_lambda_optimization_satisfies_suyu_stationarity(kind):
    source_grid, observation, operator = make_identity_problem()
    result = optimize_regularization(
        observation,
        source_grid,
        operator,
        kind,
        initial_log_strength=np.log(10.0),
        log_strength_bounds=(-8.0, 15.0),
    )
    assert result.success
    assert 0.0 < result.effective_source_parameters < source_grid.size
    assert abs(result.stationarity_residual) < 2e-5
    assert result.reconstruction.log_evidence == pytest.approx(
        result.log_evidence, rel=1e-10
    )
    step = 0.2
    regularization = regularization_matrix(source_grid, kind)
    data = observation.data.reshape(-1)
    noise = observation.noise_map.reshape(-1)
    lower = float(
        gaussian_source_log_evidence(
            data, noise, operator, regularization, result.log_strength - step
        )[0]
    )
    upper = float(
        gaussian_source_log_evidence(
            data, noise, operator, regularization, result.log_strength + step
        )[0]
    )
    assert result.log_evidence > lower
    assert result.log_evidence > upper


def test_invalid_log_strength_setup_is_rejected():
    source_grid, observation, operator = make_identity_problem()
    with pytest.raises(ValueError, match="strictly within"):
        optimize_regularization(
            observation,
            source_grid,
            operator,
            "zeroth",
            initial_log_strength=20.0,
            log_strength_bounds=(-5.0, 5.0),
        )

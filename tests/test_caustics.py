import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from akira import (
    CartesianGrid,
    LensingCurves,
    LensPlane,
    SIS,
    find_lensing_curves,
    lens_mapping_determinant,
    plot_caustics,
)


def test_sis_determinant_and_curves_have_expected_radii() -> None:
    einstein_radius = 0.8
    lens = LensPlane((SIS(einstein_radius),))
    determinant = lens_mapping_determinant(lens, np.array([[1.6, 0.0]]))
    np.testing.assert_allclose(determinant, 0.5, atol=1e-12)

    search_grid = CartesianGrid((241, 241), pixel_scale=0.01)
    curves = find_lensing_curves(lens, search_grid)

    assert len(curves.critical_curves) == 1
    critical_radius = np.linalg.norm(curves.critical_curves[0], axis=1)
    caustic_radius = np.linalg.norm(curves.caustics[0], axis=1)
    np.testing.assert_allclose(critical_radius, einstein_radius, atol=2e-4)
    assert np.max(caustic_radius) < 2e-4


def test_plot_caustics_adds_each_curve_and_only_one_legend_label() -> None:
    curves = LensingCurves(
        critical_curves=(),
        caustics=(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 1.0]]),
        ),
    )
    figure, axis = plt.subplots()
    artists = plot_caustics(axis, curves, color="white")

    assert len(artists) == 2
    assert artists[0].get_label() == "caustic"
    assert artists[1].get_label() == "_nolegend_"
    assert artists[0].get_color() == "white"
    plt.close(figure)

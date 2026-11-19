"""Run the first verified Project Akira lens simulation."""

import numpy as np

from akira import CartesianGrid, Gaussian, SIS, render_sis_gaussian


def main() -> None:
    grid = CartesianGrid(shape=(200, 200), pixel_scale=0.02, supersampling=2)
    lens = SIS(einstein_radius=1.0)
    source = Gaussian(intensity=1.0, sigma=0.05)

    image = np.asarray(render_sis_gaussian(grid, lens, source))
    coordinates = np.asarray(grid.coordinates())
    radii = np.linalg.norm(coordinates, axis=-1)
    peak_radius = radii.reshape(-1)[np.argmax(image)]

    print(f"image shape: {image.shape}")
    print(f"image dtype: {image.dtype}")
    print(f"peak Einstein-ring radius: {peak_radius:.6f} arcsec")
    print(f"expected radius: {lens.einstein_radius:.6f} arcsec")


if __name__ == "__main__":
    main()


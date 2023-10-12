from typing import Tuple

import einops
import numpy as np
from scipy.interpolate import splprep, splev

from morphosamplers import Path
from morphosamplers.core import MorphoSampler
from morphosamplers.sample_types import Points


class PointSampler(MorphoSampler):
    spacing: float

    def sample(self, obj: Path) -> Points:
        """Sample a `Path` to produces an `(n, 3)` array of points."""
        tck, total_length = self.prepare_spline(obj)
        n_samples = total_length // self.spacing
        max_u = 1 - ((total_length % self.spacing) / total_length)
        u = np.linspace(0, max_u, num=int(n_samples))
        return einops.rearrange(splev(u, tck), 'xyz b -> b xyz')

    @staticmethod
    def prepare_spline(
        path: Path, n_initial_samples: int = 10_000
    ) -> Tuple[Tuple[np.ndarray], float]:
        # oversample an initial spline between control points
        points = einops.rearrange(path.control_points, 'b xyz -> xyz b')
        u = np.linspace(0, 1, num=n_initial_samples)
        spline_order = 3 if len(path) > 3 else len(path) - 1
        tck, _ = splprep(points, s=0, k=spline_order)
        samples = einops.rearrange(splev(u, tck), 'xyz b -> b xyz')

        # calculate the cumulative length as we move along the path.
        inter_point_differences = np.diff(samples, axis=0)
        inter_point_distances = np.linalg.norm(inter_point_differences, axis=1)
        cumulative_distance = np.cumsum(inter_point_distances)
        total_length = float(cumulative_distance[-1])
        cumulative_distance /= total_length  # normalise to unit length
        cumulative_distance = np.insert(cumulative_distance, 0, 0)

        # equidistant samples in u yield equidistant samples in euclidean space
        tck, _ = splprep(samples.T, u=cumulative_distance, s=0, k=spline_order)
        return tck, total_length

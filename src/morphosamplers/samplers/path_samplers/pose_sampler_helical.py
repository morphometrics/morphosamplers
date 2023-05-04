import numpy as np
from scipy.spatial.transform import Rotation as R

from morphosamplers.core import MorphoSampler
from morphosamplers.models import Path
from morphosamplers.sample_types import PoseSet

from .pose_sampler_parallel import PoseSampler


class HelicalPoseSampler(MorphoSampler):
    """Sample poses along the backbone of a helical path.

    In the sample poses
    - positions will be separated by `spacing`.
    - z-axis at each position will be aligned with the filament axis
    - xy-planes will be rotated by `twist` degrees between adjacent positions.
    """
    spacing: float  # spacing between positions
    twist: float  # helical twist per subunit in degrees

    def sample(self, obj: Path) -> PoseSet:
        # get parallel poses along filament axis
        sampler = PoseSampler(spacing=self.spacing)
        poses = sampler.sample(obj)

        # rotate poses so they respect helical parameters
        angles = np.linspace(start=0, stop=self.twist * len(poses), num=len(poses))
        Rz = R.from_euler('z', angles=angles, degrees=True).as_matrix()
        new_orientations = poses.orientations @ Rz
        return PoseSet(positions=poses.positions, orientations=new_orientations)

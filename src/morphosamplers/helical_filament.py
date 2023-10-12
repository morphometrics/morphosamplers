from scipy.spatial.transform import Rotation
import numpy as np

from .spline import Spline3D


class HelicalFilament(Spline3D):

    def sample_helical(self, rise, twist=0, radial_offset=0, cyclic_symmetry_order=1, twist_offset=0, degrees=True):
        positions = []
        orientations = []
        base_positions = self.sample(separation=rise)
        base_orientations = self.sample_orientations(separation=rise)

        # twist around the z acis by the given angle
        twist_angles = twist_offset + (np.arange(len(base_positions)) * twist)
        Rz = Rotation.from_euler('z', angles=twist_angles, degrees=degrees)
        base_orientations *= Rz

        # create symmetry duplicates around the axis of the filament
        for i in range(cyclic_symmetry_order):
            # rotate around z by i * 360/c-symmetry-group
            twist_angles = i * (2 * np.pi / cyclic_symmetry_order)
            Rz = Rotation.from_euler('z', angles=twist_angles)
            ori = base_orientations * Rz
            # shift all particles away from the filament axis by self.radius
            # direction of shift is determined by the twist of the particle (along its y axis)
            y = ori.as_matrix()[..., 1]
            pos = base_positions + y * radial_offset
            positions.append(pos)
            orientations.append(ori)

        return np.concatenate(positions, axis=0), np.concatenate(orientations, axis=0)

    @staticmethod
    def subunits_per_turn(self, twist):
        """Number of subunits per full helical turn."""
        if twist == 0:
            return 1.0
        return 360 / np.abs(twist)

    @staticmethod
    def pitch(self, twist, rise):
        """Helical pitch, the vertical distance along the helix for one full turn."""
        if twist == 0:
            return rise
        return 360 * rise / np.abs(twist)

    @staticmethod
    def handedness(self, twist):
        if twist == 0:
            return 1
        return np.sign(twist)

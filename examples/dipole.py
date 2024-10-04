import napari
import numpy as np
import einops

from morphosamplers import Dipole, dipole_samplers
from morphosamplers.samplers.dipole_samplers import PoseSampler

# dipole
dipole = Dipole(center=np.array([[0.0, 0.0, 0.0]]),
                direction=np.array([[0.0, 0.0, 10.0]]))

# sample points within the circle with hexagonal packing
disk_sampler = dipole_samplers.DiskSampler(spacing=8, radius=50)
disk_positions = disk_sampler.sample(dipole)

# poses
pose_sampler = PoseSampler()
poses = pose_sampler.sample(dipole)

# create orientations for each of the disk points
n_disk_points_per_dipole = disk_positions.shape[0] // len(dipole.center)
orientations = einops.repeat(poses.orientations, 'b i j -> (b repeat) i j', repeat=n_disk_points_per_dipole)

### visualise
viewer = napari.Viewer(ndisplay=3)

# positions
viewer.add_points(disk_positions, size=5, name='hexagonal points')

# poses
viewer.add_vectors(
    data=np.stack(
        [disk_positions, orientations[:, :, 2]],
        axis=1
    ),
    length=10,
    edge_color='orange',
    name='dipole pose z'
)


viewer.add_vectors(
    np.stack(
        [disk_positions, orientations[:, :, 1]],
        axis=1
    ),
    length=10,
    edge_color='cornflowerblue',
    name='dipole pose y'
)

napari.run()

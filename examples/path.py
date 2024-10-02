import numpy as np
import napari

from morphosamplers import Path, path_samplers
from morphosamplers.samplers.path_samplers.pose_sampler_rings import RingPoseSampler

# create some control points for a path
xyz = np.linspace([0, 0, 0], [0, 0, 200], num=10)
xyz[:, :2] += np.random.normal(loc=0, scale=10, size=(10, 2))

# make a Path model from these control points
path = Path(control_points=xyz)

### create different types of samplers and use them to sample the path
# points
point_sampler = path_samplers.PointSampler(spacing=10)
positions = point_sampler.sample(path)  # equally spaced positions

# poses
pose_sampler = path_samplers.PoseSampler(spacing=10)
parallel_poses = pose_sampler.sample(path)  # equally spaces parallel poses

# helical poses
helical_pose_sampler = path_samplers.HelicalPoseSampler(spacing=10, twist=30)
helical_poses = helical_pose_sampler.sample(path)  # poses related by 'twist' degrees

# ring poses
ring_pose_sampler = RingPoseSampler(spacing=30, n_points_per_ring=10, radius=20)
ring_poses = ring_pose_sampler.sample(path)     # poses on the ring pointing outwards

### visualise
viewer = napari.Viewer(ndisplay=3)

# positions
viewer.add_points(positions, size=5)
viewer.add_points(ring_poses.positions, size=2, name='ring positions')

# parallel poses
viewer.add_vectors(
    data=np.stack(
        [parallel_poses.positions, parallel_poses.orientations[:, :, 2]],
        axis=1
    ),
    length=5,
    edge_color='orange',
    name='parallel pose z'
)
viewer.add_vectors(
    np.stack(
        [parallel_poses.positions, parallel_poses.orientations[:, :, 1]],
        axis=1
    ),
    length=2,
    edge_color='cornflowerblue',
    name='parallel pose y',
)


# helical poses
viewer.add_vectors(
    data=np.stack(
        [helical_poses.positions, helical_poses.orientations[:, :, 2]],
        axis=1
    ),
    length=5,
    edge_color='orange',
    name='helical pose z'
)
viewer.add_vectors(
    np.stack(
        [helical_poses.positions, helical_poses.orientations[:, :, 1]],
        axis=1
    ),
    length=2,
    edge_color='cornflowerblue',
    name='helical pose y',
)


# ring poses
viewer.add_vectors(
    data=np.stack(
        [ring_poses.positions, ring_poses.orientations[:, :, 2]],
        axis=1
    ),
    length=5,
    edge_color='orange',
    name='ring pose z'
)
viewer.add_vectors(
    np.stack(
        [ring_poses.positions, ring_poses.orientations[:, :, 1]],
        axis=1
    ),
    length=2,
    edge_color='cornflowerblue',
    name='ring pose y',
)

# plot the rings
n_rings = len(ring_poses.positions) // ring_pose_sampler.n_points_per_ring

for ring_index in range(n_rings):
    ring_points = ring_poses.positions[ring_index::n_rings]
    viewer.add_shapes(ring_points, shape_type='polygon', edge_color='yellow',
                      face_color='transparent', name=f'ring {ring_index}')

# run napari
napari.run()

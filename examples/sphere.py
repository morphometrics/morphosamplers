import napari
import numpy as np

from morphosamplers import Sphere, sphere_samplers

# create a sphere model
sphere = MorphoModels.Sphere(center=(5, 5, 5), radius=20)

# create different types of samplers and use them to sample the sphere
# points
point_sampler = sphere_samplers.PointSampler(spacing=2)
points = point_sampler.sample(sphere)

# poses, z vector normal to the sphere
pose_sampler = sphere_samplers.PoseSampler(spacing=2)
poses = pose_sampler.sample(sphere)


# visualise
viewer = napari.Viewer(ndisplay=3)
viewer.add_points(poses.positions)
viewer.add_vectors(
    data=np.stack(
        [poses.positions, poses.orientations[:, :, 2]],
        axis=1
    ),
    length=5,
    edge_color='orange',
    name='pose z'
)
viewer.add_vectors(
    np.stack(
        [poses.positions, poses.orientations[:, :, 1]],
        axis=1
    ),
    length=2,
    edge_color='cornflowerblue',
    name='pose y',
)
napari.run()
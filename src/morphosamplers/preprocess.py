import numpy as np
import pandas as pd
from skimage import morphology
import skan
from skan.csr import summarize
import dijkstra3d
from itertools import product
from functools import partial


def subsample_inclusive(lst, sampling_step=10):
    """Subsample a list retaining the extrema."""
    subsampled = list(lst[::sampling_step])

    if len(lst) % sampling_step:
        subsampled.pop(-1)
        subsampled.append(lst[-1])
    return subsampled


def dist_euclidean(a, b):
    """Euclidean distance between to points."""
    return np.linalg.norm(b - a)


def dist_dijkstra(a, b, field):
    """Geodesic distance between to points inside a boolean label."""
    path = dijkstra3d.binary_dijkstra(field, a.astype(int), b.astype(int), background_color=0).astype(int)
    return np.linalg.norm(path[1:] - path[:-1], axis=1).sum()


def connect_paths(paths, dist_func):
    """Connect disjointed paths by minimizing distance between connected points."""
    source = paths[0]
    targets = paths[1:]

    while targets:
        # TODO: optimize by reusing already computed distances and also by using the dijkstra3d
        #       tools for reusing precomputed distance fields
        dists = {}
        for idx, target in enumerate(targets):
            for s_idx, t_idx in product((0, -1), (0, -1)):
                start = source[s_idx]
                end = target[t_idx]
                dists[(idx, s_idx, t_idx)] = dist_func(end, start)

        closest_idx, s_idx, t_idx = sorted(dists.items(), key=lambda item: item[1])[0][0]
        closest = targets.pop(closest_idx)
        # remove dists to/from the removed target (Combine with reusing distances)
        # dists = {k: v for k, v in dists.items() if k[0] != closest_idx and k[1] != s_idx and k[2] != t_idx}

        if t_idx == -1:
            closest = np.flipud(closest)
        if s_idx == 0:
            source = np.flipud(source)
        source = np.concatenate([source, closest])

    return source


def get_label_paths_2d(slice_label: np.ndarray, sampling_step=10):
    """Extract linear paths from a 2D label.""" 
    if np.sum(slice_label) <= 1:
        return []

    # skeletonize the label
    skeletonized_seg = morphology.skeletonize(slice_label)
    try:
        skel = skan.Skeleton(skeletonized_seg, spacing=1, source_image=slice_label)
    except ValueError as e:
        if 'index pointer size' in e.args[0]:
            return []
        else:
            raise

    # summarize the skeleton to get the main skeleton components
    summary = summarize(skel, find_main_branch=True)

    # get the main path indices and the corresponding skeleton ids
    main_path_idcs = np.argwhere(summary["main"]).flatten()
    skeleton_ids = summary["skeleton-id"][main_path_idcs]

    # get the main paths and subsample them
    main_paths = [skel.paths_list()[idx] for idx in main_path_idcs]
    main_paths = [subsample_inclusive(path, sampling_step) for path in main_paths]

    # connect the main paths
    coords = [skel.coordinates[path] for path in main_paths]

    # create a dictionary with the skeleton ids as keys and the corresponding coordinates as values
    connected_component_coords = {sk_id: [] for sk_id in skeleton_ids}
    for k, sk_id in enumerate(skeleton_ids):
        connected_component_coords[sk_id].append(coords[k])

    connected = [connect_paths(paths, dist_euclidean) for paths in connected_component_coords.values()]
    return [pd.DataFrame(path).drop_duplicates().to_numpy() for path in connected]


def get_label_paths_3d(labels_data, axis=0, slicing_step=20, sampling_step=20):
    """Extract linear paths for surface generation from a 3D segmentation""" 
    connected_components, n_components = morphology.label(labels_data, return_num=True)

    surfaces_paths = []
    # different disconnected labels are processed separately
    for i in range(n_components):
        label = connected_components == i + 1
        paths = [get_label_paths_2d(sl, sampling_step) for sl in subsample_inclusive(label, slicing_step)]
        padded = []
        for i, p in enumerate(paths):
            if p:
                padded.append([np.pad(pp, ((0, 0), (1, 0)), constant_values=i * slicing_step) for pp in p])
        surfaces_paths.append([connect_paths(p, partial(dist_dijkstra, field=label)) for p in padded])
    return surfaces_paths

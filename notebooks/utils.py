import numpy as np
from scipy import spatial
from tqdm import tqdm as tqdm

import gudhi

#H = np.transpose([np.repeat(l, len(l)), np.tile(l, len(l))])
H = np.array([[-1, -1], [-1,  0], [-1,  1],
              [ 0, -1], [ 0,  0],  [ 0,  1],
              [ 1, -1], [ 1,  0], [ 1,  1]])

def get_Q():
    n_discr = 100
    theta, phi = np.linspace(-np.pi, np.pi, n_discr)[:, None], np.linspace(-np.pi, np.pi, n_discr)[None, :]
    abcd
    polynomial = lambda x: ()

def get_density_filtration_batch(points, k_th_nearest=200,
                                     percentage=0.3):
    n_points, dim = points.shape
    n = 4167
    
    # Filtration
    p, k = percentage, k_th_nearest
    kth_neighbor = [] #np.zeros(n_points)
    print("Starting loops")
    for pts in tqdm(np.array_split(points, n, axis=0)):
        print("Loop")
        distance = spatial.distance.cdist(pts, points)
        print("Computed distance")
        partition = np.partition(v, k, axis=1)[k]
        print("Computed_partition")
        kth_neighbor.append(partition)
    kth_neighbor = np.concatenate(kth_neighbor)
    
    smallest_distances_indices = np.argsort(kth_neighbor)[:int(len(kth_neighbor)*p)]
    dense_points = points[smallest_distances_indices, :]
    
    return dense_points

def get_density_filtration_efficient(points, k_th_nearest=200,
                                     percentage=0.3):
    n_points, dim = points.shape
    kth_neighbor = []
    
    p, k = percentage, k_th_nearest
    for n, p in tqdm(enumerate(points)):
        v = np.sqrt(np.sum(np.square(p-points), axis=1)) # np.linalg.norm(p-points, ord=2, axis=1)
        kth_neighbor.append(np.partition(v, k, axis=0)[k])
    kth_neighbor = np.array(kth_neighbor)
    
    smallest_distances_indices = np.argsort(kth_neighbor)[:int(len(kth_neighbor)*p)]
    dense_points = points[smallest_distances_indices, :]
    
    return dense_points

def get_density_filtration(points, k_th_nearest=200,
                           percentage=0.3):
    n_points, dim = points.shape
    
    # Filtration
    p, k = percentage, k_th_nearest
    distances = spatial.distance.squareform(spatial.distance.pdist(points))
    kth_neighbor = np.partition(distances, k+1, axis=0)[:, k+1] #(k+1), because the self is included
    smallest_distances_indices = np.argsort(kth_neighbor)[:int(len(kth_neighbor)*p)]
    dense_points = points[smallest_distances_indices, :]
    
    return dense_points

def get_complex(X):
    cmplx = gudhi.RipsComplex(points=X,).create_simplex_tree(max_dimension=1)  # Only get the 1-skeleton this time
    cmplx.collapse_edges(nb_iterations=2)
    cmplx.expansion(3)
    return cmplx

def get_persistence(cmplx):
    """Compute persistence of `cmplx` and return the diagrams in the gtda format."""
    cmplx.compute_persistence()
    h0, h1, h2 = [cmplx.persistence_intervals_in_dimension(d) for d in range(3)]
    persistence = cmplx.persistence()
    dgm = np.concatenate([np.concatenate([h, d*np.ones((h.shape[0], 1))], axis=1)
                      for h, d in zip([h0,h1,h2], range(3))], axis=0)
    return dgm
import numpy as np
from scipy import spatial
from tqdm import tqdm as tqdm

import gudhi

H = np.array([[-1, -1], [-1,  0], [-1,  1],
              [ 0, -1], [ 0,  0],  [ 0,  1],
              [ 1, -1], [ 1,  0], [ 1,  1]]).T

D = np.array([[2, -1, 0, -1, 0, 0, 0, 0, 0],
              [-1, 3, -1, 0, -1, 0, 0, 0, 0],
              [0, -1, 2, 0, 0, -1, 0, 0, 0],
              [-1, 0, 0, 3, -1, 0, -1, 0, 0],
              [0, -1, 0, -1, 4, -1, 0, -1, 0],
              [0, 0, -1, 0, -1, 3, 0, 0, -1],
              [0, 0, 0, -1, 0, 0, 2, -1, 0],
              [0, 0, 0, 0, -1, 0, -1, 3, -1],
              [0, 0, 0, 0, 0, -1, 0, -1, 2]]).astype(float)

A = np.array([[1, 0, -1, 1, 0, -1, 1, 0, -1],
              [1, 1, 1, 0, 0, 0, -1, -1, -1],
              [1, -2, 1, 1, -2, 1, 1, -2, 1],
              [1, 1, 1, -2, -2, -2, 1, 1, 1],
              [1, 0, -1, 0, 0, 0, -1, 0, 1],
              [1, 0, -1, -2, 0, 2, 1, 0, -1],
              [1, -2, 1, 0, 0, 0, -1, 2, -1],
              [1, -2, 1, -2, 4, -2, 1, -2, 1]]).T
Lambda = 1/np.diag(A.T@A)
A = A/np.sqrt(np.array([6., 6., 54., 54., 8., 48., 48., 216.]))


def polynomial_g_on_s1(theta, phi):
    ab, cd = [np.stack([np.cos(t), np.sin(t)], axis=-1).T for t in [theta, phi]]
    return polynomial_g(ab, cd)


def polynomial_g(ab, cd):
    ab_ = ab[:, :, None]
    cd_ = cd[:, :, None]
    H_ = H[:, None, :]
    return cd_[0]*(ab_[0]*H_[0] + ab_[1]*H_[1])**2 + cd_[1]*(ab_[0]*H_[0] + ab_[1]*H_[1])


def get_Q_bar(n_sample, neighborhood_size=0.00):
    u, xi = np.random.rand(n_sample, 1), np.random.choice(np.array([-2, -1, 0, 1]), size=(n_sample,1))
    theta = (np.pi/2. - 2.*neighborhood_size)*u + neighborhood_size + xi*np.pi/2.
    sign = np.random.choice(np.array([-1, 1]), size=(n_sample,))
    ab = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).T
    cd = np.stack([sign, np.zeros(sign.shape[0])], axis=-1).T
    
    q_bar = polynomial_g(ab, cd)
    q_bar_new_basis = Lambda*(A.T@q_bar.T).T
    return q_bar_new_basis


def get_Q(points, n_sample, return_indices=False):
    """For each of the `n_sample` sampled points from \bar{Q}, return the point
    which is the closest.
    """
    q_bar_new_basis = get_Q_bar(n_sample)
    closest_points = np.argmin(cdist(q_bar_new_basis, points), axis=1)
    if return_indices:
        return closest_points
    else:
        return points[closest_points, :]


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

def get_complex_from_distance_matrix(distance_matrix):
    cmplx = gudhi.RipsComplex(distance_matrix=distance_matrix).create_simplex_tree(max_dimension=1)  # Only get the 1-skeleton this time
    cmplx.collapse_edges(nb_iterations=2)
    cmplx.expansion(3)
    return cmplx

def get_persistence(cmplx):
    """Compute persistence of `cmplx` and return the diagrams in the gtda format."""
    cmplx.compute_persistence()
    h0, h1, h2 = [cmplx.persistence_intervals_in_dimension(d) for d in range(3)]
    persistence = cmplx.persistence()
    dgm = np.concatenate([np.concatenate([h, d*np.ones((h.shape[0], 1))], axis=1)
                          for h, d in zip([h0,h1,h2], range(3))
                          if h.shape[0]>0], axis=0)
    return dgm
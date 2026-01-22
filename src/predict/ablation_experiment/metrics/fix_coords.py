import numpy as np


def fix_coords_by_xyz(x: np.ndarray) -> np.ndarray:
    # first at [0,0,0]
    x = x - x[0, :]
    # second at [0,0,z]
    theta_z = np.arctan2(x[1, 0], x[1, 1])
    theta_x = np.arctan2(np.sqrt(x[1, 0] ** 2 + x[1, 1] ** 2), x[1, 2])
    rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_mat = np.matmul(rx, rz)
    x = np.matmul(rot_mat, x.T).T
    # third at [0, y, z]
    theta_z = np.arctan2(x[2, 0], x[2, 1])
    rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    x = np.matmul(rz, x.T).T
    return x

def fix_coords_by_svd(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = reference.shape[0]
    # center on centroid
    cc_tar = sum(target) / n
    cc_src = sum(reference) / n
    c_target = target - cc_tar
    c_source = reference - cc_src
    # correlation matrix
    a = c_target.T @ c_source
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    rot = (vt.T @ u.T).T
    rot = rot * np.linalg.det(rot)
    if np.linalg.det(rot) < 0:
        vt[2, :] *= -1
        rot = (vt.T @ u.T).T
    tran = cc_src - (cc_tar @ rot)
    # transform
    transformed_tar = (target @ rot) + tran
    
    return transformed_tar

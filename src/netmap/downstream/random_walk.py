"""Generic random-walk-with-restart (RWR) library.

Graph-agnostic: operates on any square weighted adjacency matrix and has no
knowledge of genes, celltypes, or ``grn_adata``. Callers assemble the graph and
pass it in; this module only implements the walk itself.
"""

import numpy as np
import scipy.sparse as scs


def random_walk_with_restart(adjacency, seed_vec, restart_prob=0.2, tol=1e-6, max_iter=200):
    """Run a random walk with restart (personalized PageRank) to convergence.

    Iterates ``p_{t+1} = (1 - r) * (W^T p_t + dangling_correction) + r * p0``
    where ``W`` is the row-normalized transition matrix derived from
    ``adjacency`` and ``p0`` is the (normalized) ``seed_vec``. Nodes with zero
    out-degree ("dangling" nodes) redistribute their probability mass via
    ``seed_vec`` each iteration instead of letting it vanish, matching the
    standard PageRank dangling-node fix.

    Args:
        adjacency (scipy.sparse matrix): Square, non-negative weighted
            adjacency matrix, shape ``(n, n)``. ``adjacency[i, j]`` is the
            weight of the edge from node ``i`` to node ``j``.
        seed_vec (np.ndarray): Restart/seed distribution, shape ``(n,)``.
            Renormalized to sum to 1 internally.
        restart_prob (float): Restart probability ``r`` in ``(0, 1]``.
            Defaults to 0.2.
        tol (float): Convergence tolerance on the L1 change between
            successive iterates. Defaults to 1e-6.
        max_iter (int): Maximum number of iterations. Defaults to 200.

    Returns:
        np.ndarray: Converged probability distribution over all ``n`` nodes,
        shape ``(n,)``, summing to 1.
    """
    adjacency = scs.csr_matrix(adjacency)
    n = adjacency.shape[0]

    seed_vec = np.asarray(seed_vec, dtype=float).flatten()
    seed_sum = seed_vec.sum()
    if seed_sum <= 0:
        raise ValueError("seed_vec must have positive total mass.")
    seed_vec = seed_vec / seed_sum

    row_sums = np.asarray(adjacency.sum(axis=1)).flatten()
    dangling_mask = row_sums == 0
    inv_row_sums = np.zeros(n)
    inv_row_sums[~dangling_mask] = 1.0 / row_sums[~dangling_mask]

    adjacency_t = adjacency.transpose().tocsr()

    p = seed_vec.copy()
    for _ in range(max_iter):
        propagated = adjacency_t @ (p * inv_row_sums)
        dangling_mass = p[dangling_mask].sum()
        p_next = (1 - restart_prob) * (propagated + dangling_mass * seed_vec) + restart_prob * seed_vec
        diff = np.abs(p_next - p).sum()
        p = p_next
        if diff < tol:
            break

    return p

"""Random-walk marker-gene celltype classifier (simplified CellWalker2-style).

For each cell, assembles a graph from that cell's GRN (``grn_adata``) plus a
static bipartite layer connecting celltypes to their marker genes, then runs a
random walk with restart (:func:`netmap.downstream.random_walk.random_walk_with_restart`)
seeded at the cell's own regulon activity. The converged probability mass on
celltype nodes, renormalized, is that cell's celltype probability vector.

Edge weights are taken as ``abs(weight)`` throughout: GRN edges are directed and
signed (activation/repression), but for this walk both signs simply indicate
"these genes are functionally linked" and are treated identically.
"""

import logging

import numpy as np
import scipy.sparse as scs
from joblib import Parallel, delayed

from netmap.downstream.random_walk import random_walk_with_restart

logger = logging.getLogger(__name__)


def build_gene_index(var):
    """Build a fixed gene universe and per-edge integer index arrays.

    Args:
        var (pd.DataFrame): ``grn_adata.var``, with ``source`` and ``target``
            gene-name columns (one row per directed edge).

    Returns:
        tuple:
            - gene_to_idx (dict): Mapping from gene name to integer index.
            - src_idx (np.ndarray): Integer source-gene index per edge,
              shape ``(n_edges,)``.
            - tgt_idx (np.ndarray): Integer target-gene index per edge,
              shape ``(n_edges,)``.
    """
    genes = sorted(set(var["source"]) | set(var["target"]))
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    src_idx = var["source"].map(gene_to_idx).to_numpy()
    tgt_idx = var["target"].map(gene_to_idx).to_numpy()
    return gene_to_idx, src_idx, tgt_idx


def build_marker_incidence(gene_to_idx, marker_dict):
    """Build the static gene-celltype bipartite incidence matrix.

    Args:
        gene_to_idx (dict): Mapping from gene name to integer index, as
            returned by :func:`build_gene_index`.
        marker_dict (dict): Mapping ``{celltype: [gene, ...]}`` of marker
            genes per celltype. Not built or validated by this module — the
            caller supplies it.

    Returns:
        tuple:
            - B (scipy.sparse matrix): Incidence matrix, shape
              ``(n_genes, n_celltypes)``, with ``B[g, c] = 1`` if gene ``g``
              is a marker of celltype ``c``.
            - celltypes (list): Celltype labels, in the column order of ``B``
              (the key order of ``marker_dict``).
    """
    celltypes = list(marker_dict.keys())
    n_genes = len(gene_to_idx)

    rows, cols = [], []
    n_dropped = 0
    for ct_idx, ct in enumerate(celltypes):
        for gene in marker_dict[ct]:
            gene_idx = gene_to_idx.get(gene)
            if gene_idx is None:
                n_dropped += 1
                continue
            rows.append(gene_idx)
            cols.append(ct_idx)

    if n_dropped:
        logger.warning(
            "Dropped %d marker gene(s) not present in the GRN gene universe.", n_dropped
        )

    data = np.ones(len(rows))
    b_matrix = scs.coo_matrix((data, (rows, cols)), shape=(n_genes, len(celltypes))).tocsr()
    return b_matrix, celltypes


def compute_seed_activity(X, src_idx, tgt_idx, n_genes):
    """Compute per-cell, per-gene regulon activity as an RWR seed vector.

    For every cell, sums ``abs(weight)`` over all edges touching each gene
    (as source or target), vectorized across all cells at once via sparse
    matrix multiplication rather than a per-cell loop.

    Args:
        X: ``grn_adata.X``, shape ``(n_cells, n_edges)``. May be dense
            (``np.ndarray``) or sparse (``scipy.sparse``) — both are handled.
        src_idx (np.ndarray): Integer source-gene index per edge.
        tgt_idx (np.ndarray): Integer target-gene index per edge.
        n_genes (int): Size of the gene universe.

    Returns:
        np.ndarray: Per-cell gene activity, shape ``(n_cells, n_genes)``, each
        row normalized to sum to 1. Cells with all-zero GRN rows fall back to
        a uniform distribution over genes.
    """
    n_cells, n_edges = X.shape

    x_abs = abs(X) if scs.issparse(X) else np.abs(X)

    indicator_src = scs.coo_matrix(
        (np.ones(n_edges), (np.arange(n_edges), src_idx)), shape=(n_edges, n_genes)
    ).tocsr()
    indicator_tgt = scs.coo_matrix(
        (np.ones(n_edges), (np.arange(n_edges), tgt_idx)), shape=(n_edges, n_genes)
    ).tocsr()

    activity = indicator_src.transpose().dot(x_abs.transpose()) + indicator_tgt.transpose().dot(
        x_abs.transpose()
    )
    activity = np.asarray(activity.todense()) if scs.issparse(activity) else np.asarray(activity)
    activity = activity.T  # (n_cells, n_genes)

    row_sums = activity.sum(axis=1)
    zero_mask = row_sums == 0
    if zero_mask.any():
        logger.warning(
            "%d cell(s) had all-zero GRN activity; falling back to a uniform seed.",
            int(zero_mask.sum()),
        )

    safe_row_sums = np.where(zero_mask, 1.0, row_sums)
    normalized = activity / safe_row_sums[:, None]
    normalized[zero_mask] = 1.0 / n_genes
    return normalized


def assemble_cell_adjacency(row_weights, src_idx, tgt_idx, n_genes, b_matrix):
    """Assemble one cell's joint gene+celltype adjacency matrix.

    Combines that cell's GRN edges (gene-gene block) with the static
    gene-celltype bipartite layer (``b_matrix`` / ``b_matrix.T``). The
    celltype-celltype block is all zero — in this simplified model, celltypes
    are connected to the graph only via their marker genes.

    Args:
        row_weights (np.ndarray): One cell's edge weights, shape
            ``(n_edges,)`` (a row of ``grn_adata.X``).
        src_idx (np.ndarray): Integer source-gene index per edge.
        tgt_idx (np.ndarray): Integer target-gene index per edge.
        n_genes (int): Size of the gene universe.
        b_matrix (scipy.sparse matrix): Gene-celltype incidence matrix, shape
            ``(n_genes, n_celltypes)``, as returned by
            :func:`build_marker_incidence`.

    Returns:
        scipy.sparse matrix: Joint adjacency matrix, shape
        ``(n_genes + n_celltypes, n_genes + n_celltypes)``, gene nodes first
        then celltype nodes.
    """
    n_celltypes = b_matrix.shape[1]
    weights = np.abs(np.asarray(row_weights).flatten())

    gene_gene = scs.coo_matrix((weights, (src_idx, tgt_idx)), shape=(n_genes, n_genes))
    celltype_celltype = scs.csr_matrix((n_celltypes, n_celltypes))

    adjacency = scs.bmat(
        [[gene_gene, b_matrix], [b_matrix.transpose(), celltype_celltype]], format="csr"
    )
    return adjacency


def run_celltype_random_walk(
    grn_adata, marker_dict, restart_prob=0.2, tol=1e-6, max_iter=200, n_jobs=-1
):
    """Run the per-cell celltype random walk over an entire ``grn_adata``.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData (``obs``=cells, ``var``=edges
            with ``source``/``target`` columns, ``X``=edge weights). ``X``
            must already be populated (e.g. via ``retrieve_top_edges`` /
            ``retrieve_edges_by_index``), not Parquet-backed placeholder.
        marker_dict (dict): Mapping ``{celltype: [gene, ...]}`` of marker
            genes per celltype, supplied by the caller.
        restart_prob (float): RWR restart probability. Defaults to 0.2.
        tol (float): RWR convergence tolerance. Defaults to 1e-6.
        max_iter (int): RWR maximum iterations. Defaults to 200.
        n_jobs (int): Number of parallel jobs (``joblib.Parallel`` semantics,
            ``-1`` uses all cores). Cells are independent, so this is
            embarrassingly parallel. Defaults to -1.

    Returns:
        tuple:
            - proba (np.ndarray): Celltype probability matrix, shape
              ``(n_cells, n_celltypes)``, each row summing to 1.
            - celltypes (list): Celltype labels, in the column order of
              ``proba``.
    """
    var = grn_adata.var
    gene_to_idx, src_idx, tgt_idx = build_gene_index(var)
    n_genes = len(gene_to_idx)

    b_matrix, celltypes = build_marker_incidence(gene_to_idx, marker_dict)
    n_celltypes = len(celltypes)

    x = grn_adata.X
    seed_activity = compute_seed_activity(x, src_idx, tgt_idx, n_genes)

    def _get_row(i):
        row = x[i]
        if scs.issparse(row):
            row = np.asarray(row.todense()).flatten()
        else:
            row = np.asarray(row).flatten()
        return row

    def _one_cell(i):
        row_weights = _get_row(i)
        adjacency = assemble_cell_adjacency(row_weights, src_idx, tgt_idx, n_genes, b_matrix)
        seed_vec = np.concatenate([seed_activity[i], np.zeros(n_celltypes)])

        p = random_walk_with_restart(
            adjacency, seed_vec, restart_prob=restart_prob, tol=tol, max_iter=max_iter
        )

        celltype_proba = p[n_genes:]
        total = celltype_proba.sum()
        if total > 0:
            celltype_proba = celltype_proba / total
        else:
            celltype_proba = np.full(n_celltypes, 1.0 / n_celltypes)
        return celltype_proba

    n_cells = grn_adata.n_obs
    results = Parallel(n_jobs=n_jobs)(delayed(_one_cell)(i) for i in range(n_cells))
    proba = np.vstack(results)
    return proba, celltypes

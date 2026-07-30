
"""Random-walk marker-gene celltype classifier (simplified CellWalker2-style).

For each cell, assembles a graph from that cell's GRN (``grn_adata``) plus a
static bipartite layer connecting celltypes to their marker genes, then runs a
random walk with restart (:func:`netmap.downstream.random_walk.random_walk_with_restart`)
seeded at the cell's own regulon activity. The converged probability mass on
celltype nodes, renormalized, is that cell's celltype probability vector.

GRN edges are directed and signed (activation/repression). The two places
that consume edge weights treat sign differently: :func:`compute_seed_activity`
uses ``abs(weight)`` (both directions indicate regulatory activity for seeding
purposes), while :func:`assemble_cell_adjacency` clips negative weights to 0
(repressive edges don't propagate random-walk mass — they're simply absent
from that cell's graph).
"""

import logging

import numpy as np
import scipy.sparse as scs
from joblib import Parallel, delayed
import numpy as np
import scipy.sparse as scs

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




def compute_edge_idf_weights(X, count_nonzero=None, smoothing=1.0, log_scale=False):
    """Compute an inverse-document-frequency-style weight per edge.

    Edges that are nonzero in many cells (common across the dataset — likely
    generic/uninformative regulatory relationships) get down-weighted; edges
    nonzero in only a few cells (rare, likely cell-type-specific) keep close
    to full weight. Meant to always be computed from the *raw*, unsmoothed
    edge-weight matrix — running this after :func:`smooth_grn_with_knn` would
    inflate the counts, since smoothing spreads nonzero values to neighboring
    cells that didn't originally have that edge.

    Args:
        X: The raw ``grn_adata.X``, shape ``(n_cells, n_edges)``. Only used
            to derive ``count_nonzero`` when it isn't supplied directly.
        count_nonzero (np.ndarray, optional): Precomputed per-edge count of
            cells with a nonzero weight, shape ``(n_edges,)`` — e.g.
            ``grn_adata.var['count_nonzero'].to_numpy()``, which
            ``retrieve_top_edges``/``retrieve_edges_by_index`` already
            populate. If ``None``, derived from ``X`` directly.
        smoothing (float): Added to ``count_nonzero`` before dividing, to
            avoid division by zero and cap the multiplier for edges present
            in only a handful of cells. Defaults to 1.0.
        log_scale (bool): If ``True``, use
            ``log(n_cells / (count_nonzero + smoothing))`` (classic log-IDF,
            compressed range) instead of the raw ratio
            ``n_cells / (count_nonzero + smoothing)``. Defaults to ``False``
            — weight decreases directly proportional to how often the edge
            appears, as opposed to logarithmically.

    Returns:
        np.ndarray: Per-edge IDF weight, shape ``(n_edges,)``.
    """
    n_cells = X.shape[0]
    if count_nonzero is None:
        if scs.issparse(X):
            count_nonzero = np.asarray((X != 0).sum(axis=0)).flatten()
        else:
            count_nonzero = np.count_nonzero(X, axis=0)
    else:
        count_nonzero = np.asarray(count_nonzero).flatten()

    idf = n_cells / (count_nonzero + smoothing)
    if log_scale:
        idf = np.log(idf)
    return idf


def apply_edge_idf(X, idf):
    """Rescale each edge's weight by its IDF weight, broadcasting over cells.

    Args:
        X: ``grn_adata.X`` (or an already kNN-smoothed version of it), shape
            ``(n_cells, n_edges)``. Dense or sparse.
        idf (np.ndarray): Per-edge weight, shape ``(n_edges,)``, as returned
            by :func:`compute_edge_idf_weights`.

    Returns:
        Reweighted edge-weight matrix, same shape and sparse/dense-ness as
        ``X``.
    """
    if scs.issparse(X):
        return X.multiply(idf[None, :]).tocsr()
    return X * idf[None, :]


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


def assemble_cell_adjacency(row_weights, src_idx, tgt_idx, n_genes, b_matrix, top_k=None):
    """Assemble one cell's joint gene+celltype adjacency matrix.

    Combines that cell's GRN edges (gene-gene block) with the static
    gene-celltype bipartite layer (``b_matrix`` / ``b_matrix.T``). The
    celltype-celltype block is all zero — in this simplified model, celltypes
    are connected to the graph only via their marker genes.

    Negative edge weights (repression) are clipped to 0 rather than taken as
    ``abs(weight)`` — repressive edges don't propagate random-walk mass in
    this model, they're simply absent from the graph for that cell.

    Args:
        row_weights (np.ndarray): One cell's edge weights, shape
            ``(n_edges,)`` (a row of ``grn_adata.X``).
        src_idx (np.ndarray): Integer source-gene index per edge.
        tgt_idx (np.ndarray): Integer target-gene index per edge.
        n_genes (int): Size of the gene universe.
        b_matrix (scipy.sparse matrix): Gene-celltype incidence matrix, shape
            ``(n_genes, n_celltypes)``, as returned by
            :func:`build_marker_incidence`.
        top_k (int, optional): If set, keep only the ``top_k`` highest-weight
            edges for this cell (by the clipped weight) and drop the rest
            entirely from the gene-gene block, instead of using all edges.
            ``None`` (default) keeps every edge.

    Returns:
        scipy.sparse matrix: Joint adjacency matrix, shape
        ``(n_genes + n_celltypes, n_genes + n_celltypes)``, gene nodes first
        then celltype nodes.
    """
    n_celltypes = b_matrix.shape[1]
    weights = np.asarray(row_weights).flatten().clip(min=0)
    edge_src, edge_tgt = src_idx, tgt_idx


    gene_gene = scs.coo_matrix((weights, (edge_src, edge_tgt)), shape=(n_genes, n_genes))
    celltype_celltype = scs.csr_matrix((n_celltypes, n_celltypes))

    adjacency = scs.bmat(
        [[gene_gene, b_matrix], [b_matrix.transpose(), celltype_celltype]], format="csr"
    )
    return adjacency


def run_celltype_random_walk(
    grn_adata,
    marker_dict,
    restart_prob=0.2,
    tol=1e-6,
    max_iter=200,
    n_jobs=-1,
    edge_idf=False,
    idf_smoothing=1.0,
    idf_log_scale=False,
    scale_by_marker_count=False,
    top_genes_k=None,
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
        edge_idf (bool): If ``True``, rescale edge weights by an
            inverse-document-frequency-style weight (see
            :func:`compute_edge_idf_weights`) before seed-activity and
            adjacency construction — edges common across many cells are
            down-weighted, rare/cell-type-specific edges keep their weight.
            Computed from the raw ``grn_adata.X`` (and
            ``grn_adata.var['count_nonzero']`` when present) regardless of
            whether ``cell_knn`` smoothing is also enabled. Defaults to
            ``False``.
        idf_smoothing (float): Passed through to
            :func:`compute_edge_idf_weights` when ``edge_idf`` is set.
            Defaults to 1.0.
        idf_log_scale (bool): Passed through to
            :func:`compute_edge_idf_weights` when ``edge_idf`` is set.
            Defaults to ``False``.
        scale_by_marker_count (bool): If ``True``, divide each celltype's
            converged probability mass by that celltype node's in-degree in
            ``b_matrix`` (its marker count, after gene-universe filtering)
            before the final sum-to-1 renormalization. Corrects for celltypes
            with more markers otherwise accumulating more mass purely from
            having more incoming edges, independent of actual fit to the
            cell. Defaults to ``False``.
        top_genes_k (int, optional): If set, also compute each cell's
            ``top_genes_k`` most-visited gene nodes (by that cell's own
            converged probability mass, i.e. ``p[:n_genes]`` from the same
            walk that produces its celltype probabilities — no extra walk is
            run). ``None`` (default) skips this and keeps the original
            2-tuple return signature; setting it adds a third return value.

    Returns:
        tuple:
            - proba (np.ndarray): Celltype probability matrix, shape
              ``(n_cells, n_celltypes)``, each row summing to 1.
            - celltypes (list): Celltype labels, in the column order of
              ``proba``.
            - top_genes (list, only present if ``top_genes_k`` is set): One
              entry per cell, each a list of ``(gene_name, score)`` tuples of
              length ``min(top_genes_k, n_genes)``, sorted descending by
              score.
    """
    var = grn_adata.var
    gene_to_idx, src_idx, tgt_idx = build_gene_index(var)
    n_genes = len(gene_to_idx)
    idx_to_gene = {i: g for g, i in gene_to_idx.items()} if top_genes_k is not None else None

    b_matrix, celltypes = build_marker_incidence(gene_to_idx, marker_dict)
    n_celltypes = len(celltypes)

    celltype_in_degree = np.asarray(b_matrix.sum(axis=0)).flatten()
    safe_in_degree = np.where(celltype_in_degree == 0, 1.0, celltype_in_degree)

    x = np.multiply(grn_adata.X, grn_adata.layers['mask'])


    if edge_idf:
        count_nonzero = (
            var["count_nonzero"].to_numpy() if "count_nonzero" in var.columns else None
        )
        idf = compute_edge_idf_weights(
            grn_adata.X, count_nonzero=count_nonzero, smoothing=idf_smoothing, log_scale=idf_log_scale
        )
        x = apply_edge_idf(x, idf)

    seed_activity = compute_seed_activity(x, src_idx, tgt_idx, n_genes)
    print(seed_activity)

    def _get_row(i):
        row = x[i]
        if scs.issparse(row):
            row = np.asarray(row.todense()).flatten()
        else:
            row = np.asarray(row).flatten()
        return row

    def _one_cell(i):
        row_weights = _get_row(i)
        adjacency = assemble_cell_adjacency(
            row_weights, src_idx, tgt_idx, n_genes, b_matrix,
        )
        seed_vec = np.concatenate([seed_activity[i], np.zeros(n_celltypes)])

        p = random_walk_with_restart(
            adjacency, seed_vec, restart_prob=restart_prob, tol=tol, max_iter=max_iter
        )

        celltype_proba = p[n_genes:]
        if scale_by_marker_count:
            celltype_proba = celltype_proba / safe_in_degree
        total = celltype_proba.sum()
        if total > 0:
            celltype_proba = celltype_proba / total
        else:
            celltype_proba = np.full(n_celltypes, 1.0 / n_celltypes)

        if top_genes_k is None:
            return celltype_proba

        gene_scores = p[:n_genes]
        k = min(top_genes_k, n_genes)
        top_idx = np.argpartition(gene_scores, -k)[-k:]
        top_idx = top_idx[np.argsort(gene_scores[top_idx])[::-1]]
        top_genes = [(idx_to_gene[j], float(gene_scores[j])) for j in top_idx]
        return celltype_proba, top_genes

    print('starting')
    n_cells = grn_adata.n_obs
    results = Parallel(n_jobs=n_jobs)(delayed(_one_cell)(i) for i in range(n_cells))

    if top_genes_k is None:
        proba = np.vstack(results)
        return proba, celltypes

    proba = np.vstack([r[0] for r in results])
    top_genes_per_cell = [r[1] for r in results]
    return proba, celltypes, top_genes_per_cell



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
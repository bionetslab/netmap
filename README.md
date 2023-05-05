# netmap

## Abstract classes

- InitializationWrapper; input: cell list, initial clustering or number of clusters k; output: cell clustering with k clusters
- GRNInferenceWrapper; input: expression data, cell clustering, old GRNs; output: new GRNs (one for each of the k clusters)
- CellEmbeddingWrapper; input: expression data, cluster-specific GRNs; output: low-dimensional embedding representing the GRNs (easy solution: expression of genes contained in GRNs, better solution: expressive power of GRN i yields z_i(c))
- ClusteringUpdateWrapper: input: GRN-based cell embedding, old clustering; output: new cell clustering with k clusters

## Basic implementations:
- IntializationWrapper: either user-defined cell clustering or random split into k clusters with user-provided k
- GRNInferenceWrapper: for each cluster i, infer GRN using any fast GRN inference method (e.g., ACRANCNeAP, https://github.com/bionetslab/grn-confounders/blob/main/src/confinspect/ARACNEWrapper.py, 10.1093/bioinformatics/btw216). Then, define cluster-specific GRN i as edge set of inferrred GRN i - union of edge sets of all other GRNs. Or else, just keep the top m edges of each inferred GRN i. Enforce same number of nodes/genes for each GRN. Check for GRN-wise convergence (edge-based).
- CellEmbeddingWrapper: for each cell, use only expression of genes contained in the k GRNs as low-dimenensional representation 
- ClusteringUpdateWrapper: Use spectral clustering (or any other clustering appraoch) to compute new clusters based on low-dimensional representation. Check for cluster-wise convergence.

## Termination criteria
- Cluster-wise and/or GRN-wise convergence.
- Number of iterations.
- Time limit.

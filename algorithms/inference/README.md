# AbstractGRNInference


## Example implementation of BasicGRNInference

To implement BasciGRNInference we need to implement all abstract methods from AbstractGRNInference. 

### GRN inference
Here, we simply sample some random graphs for each cluster we defined in the clustering step. We generate a number of nodes an insert an edge from A>B with probability p. This yields a directed graph for every cluster. The graphs are stored in the anndata object as a ```scipy.sparse.crs_array``` in the ```.varp``` field of the ```AnnData``` object. ```.varp``` has a flat hierarchy. To structure the information, additionally, the ```.uns``` field of the AnnData object contains a entry ```.uns['GRNs']```. For every iteration a dictionary is inserted, containing a dictionary of clusters.

This is designed intentionally so the AnnData object contains the last two iterations of the GRNs for convenience. It is recommended to remove the previous GRNs for memory efficiency.

```python
def _infer_cluster_specific_GRNS(self) -> None:
    if not 'GRNs' in self.data.uns.keys():
            self.data.uns['GRNs'] = {}

    i = self.data.uns['current_iteration']
    self.data.uns['GRNs']['iteration_'+str(i)] =  {}
    for lab in range(self.data.uns['algorithm.n_clusters']):
        self.data.uns['GRNs'][f'iteration_{i!r}'][f'cluster_{lab!r}'] = f'iteration{i!r}_cluster{lab!r}'
        index_of = np.random.choice(len(self.data.var.index), size=100, replace=False)
        binomials = np.random.binomial(n = 1, p=0.2, size=100*99)

        #initialize the sparse gene module as dok matrix and insert elements
        varp = scs.dok_array((self.data.shape[1], self.data.shape[1]))
        for elem, bin in zip(itertools.product(index_of, index_of), binomials.astype(bool)):
            if bin > 0:
                print(elem)
                varp[elem] = 1
        # transform to csr and insert fo the current iteration.
        varp = scs.csr_matrix(varp)
        self.data.varp[ f'iteration{i!r}_cluster{lab!r}'] = varp


    if i>1:
        im2 = i-2
        for GRN in self.data.uns['GRNs'][f'iteration_{im2!r}']:
            del self.data.varp[self.data.uns['GRNs'][f'iteration_{im2!r}'][GRN]]
        del self.data.uns['GRNs'][f'iteration_{im2!r}']


```

For an instance with 3 clusters would look like this after 5 iterations. ```self.data``` contains ```self.data.uns['GRNs']``` and ```self.varp``` contains the same keys in a flat hierarchy. Only the 4th and 5th iterations GRNs are stored in the data object.
```python
>>> data
AnnData object with n_obs × n_vars = 4906 × 1000
    obs: 'dataset', 'initial_clustering', 'n_counts_all'
    var: 'n_counts', 'mean', 'std'
    uns: 'log1p', 'pca', 'neighbors', 'n_clusters', 'GRNs'
    obsm: 'X_pca'
    varm: 'PCs'
    varp: 'iteration4_cluster0', 'iteration4_cluster1', 'iteration4_cluster2', 'iteration5_cluster0', 'iteration5_cluster1', 'iteration5_cluster2'
    
>>> data.uns['GRNs']
{'iteration_4': 
      {'cluster_0': 'iteration4_cluster0', 
      'cluster_1': 'iteration4_cluster1', 
      'cluster_2': 'iteration4_cluster2'}, 
 'iteration_5': 
      {'cluster_0': 'iteration5_cluster0',
      'cluster_1': 'iteration5_cluster1', 
      'cluster_2': 'iteration5_cluster2'}}
```

### Check GRN convergence
The consistency of the GRNs will be checked by comparing the old and new GRNs for each cluster. This function assumed that the cluster identity remains consistent over the iterations. Here, we implement a basic function that requires a certain fraction of edges to be consistent between the current and previous iteration.

```python
def _check_GRN_convergence(self, consistency):
    """
    This method checks if the GRNs have converged by checking the edges. For each cluster, first, the overlap between the edges in the 
    new and the old GRN are computed. For convergence it is required that the consitent edges make up a certain percentage of
    the old and the new GRN respectively. For example, the number of consitent edges is 75, and the consitency parameter is 0.75. 
    A GRN would have converged, if the total number of edges in the old and the new GRNs are both smaller or equal
    100, as 75% of the genes are identical. If the GRNs for all clusters have converged, the function returns True.

    Arguments:
    consistency: the required fraction of edges that have to be consitent between two GRNs for one cluster.
    """

    i = self.data.uns['current_iteration']
    number_of_converged_clusters = 0
    if i>=1:
        im1 = i-1
        for GRN_old, GRN_new in zip(self.data.uns['GRNs'][f'iteration_{im1!r}']):
            # multiply the matrices, which is basically an element wise and operation
            number_of_consistent_edges = int(np.sum(self.data.varp[self.data.uns['GRNs'][f'iteration_{im1!r}'][GRN_old]].multiply(
                self.data.varp[self.data.uns['GRNs'][f'iteration_{i!r}'][GRN_new]]
                )))

            if int(np.sum(self.data.varp[self.data.uns['GRNs'][f'iteration_{im1!r}'][GRN_old]])) * consistency >= number_of_consistent_edges \
                    and int(np.sum(self.data.varp[self.data.uns['GRNs'][f'iteration_{i!r}'][GRN_new]])) * consistency >= number_of_consistent_edges:
                number_of_converged_clusters = number_of_converged_clusters + 1

    # Check of thr GRNs for all clusters have converged.
    if number_of_converged_clusters == self.data.uns['algorithm.n_clusters']:
        return True
    else:
        return False
```

### Write custom results
In this case we save the final GRNs for each cluster in an ```.npz``` file.

```
def _write_results(self):

    """ 
    Write the last GNRs to file in a sparse matrix format. One file per GRN.
    """
    try:
        i = self.data.uns['current_iteration']-1
        for GRN in self.data.uns['GRNs'][f'iteration_{i!r}'].values():
            print(GRN)
            filename = op.join(self.data.uns['GNR_dir'], f'{GRN}.npz')
            scs.save_npz(file=filename, matrix=self.data.varp[GRN])
    except KeyError:
        print('Results not initialized')
    
```

### Get top k edges
Technically, we should also return the top k edges for the graph. Since we do not have edge weights here, we omit the implementation out of laziness for now.
```
def _get_top_k_edges():
    return 0
    
```

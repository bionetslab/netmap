# AbstractClustering

## BasicClustering
Basic clustering just needs to implement the new clustering update, and if required override the 

Here, we simply cluster the newly create embedding using KMeans with 2 iterations. 

The clustering represents the new cell grouping. Conforming to scanpy's usage standards, this information is stored in the ```.obs``` field of the AnnData object. We maintain three versions of the clustering throughout the algorithm execution. a) the initial clustering (kept for tracking purposes), the current clustering, and the clustering from the previous iteration to check convergence. When a new clustering is computed, the previous iteration's clustering is stored in ```self.data.obs['previous_clustering']``` and the new clustering is stored in ```self.data.obs['current_clustering']```. 


```python
def _compute_new_clustering(self) -> None:
    """
    Compute a new clustering based on the embedding computed in the previous step.
    """

    # store the old clustering
    self.data.obs['previous_clustering'] = self.data.obs['current_clustering'].copy()
    model = KMeans(n_clusters=self.data.uns['n_clusters'], random_state=0, max_iter=2)
    model.fit(self.data.obsm['embedding'])
    self.data.obs['current_clustering'] = model.labels_
```

## Clustering convergence
Here, we define cluster convergence, when at least %p of the cells retain their cluster over the consecutive runs.


## Ensure Clustering consistency.
The superclass implements the Hungarian algorithm (maximum weight matching) to ensure the clusters are consistently labelled. This functionality can be overridden, but it is not necessary.

# Implementing a custom initializer

The initializer takes care of setting all parameters, and the initial clustering labels. It may not be necessary to create a custom class for every new strategy.

By default, the initializer sets a number of parameters in the AnnData object.

```python
self.data.uns['n_clusters'] = n_clusters
self.data.uns['max_iterations'] = max_iterations
self.data.uns['current_iteration']  = 0
data.uns['GRNs'] = {}

```

## Adding an initialization strategy
For this example, we just add random labels to the cells. Note that the initial_clustering and the current_clustering initially are the same. The initial clustering is preserved, as it may be useful to compare results.
```python
def _initialize_clustering(self):
    """
    Method creates the the intial cell labels, in this case the labels
    are generated uniformly at random from the number of clusters.

    Parameters:
    -------------------------

    Returns:
    -------------------------
    A numpy array matching the number of samples in the data
    """
    if self.data is None:
        raise NotInitializerError("Data object not initialized")

    # sample uniformly at random from the number of clusters
    random_clusters = np.random.choice(self.data.uns['n_clusters'], self.data.X.shape[0])
    # add a column with the initial clustering
    self.data.obs['initial_clustering'] = random_clusters
    self.data.obs['current_clustering'] = random_clusters

```

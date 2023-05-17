# Implementing AbstractCellEmbedding

The new cell embedding is the low dimensional representation of the expression matrix based on the inferred GRNs.

## Compute new cell embedding
In  this example, the low dimensional embedding of the gene expression matrix is the subset of the matrix containing only the genes that are contained in the GRNs.

```
def _compute_new_cell_embedding(self):
    """
    Computes a new cell embedding. 
    The basic cell embedding is simply the union of all
    genes contained in all the GRNs. The embedding is stored in the 
    obsm field of the anndata object.

    """

    i = self.data.uns['current_iteration']
    all_genes = []
    # select all genes in all GRNs as the current embedding.
    for GRN in self.data.uns['GRNs'][f'iteration_{i!r}'].values():
        all_genes.append(np.unique(self.data.varp[GRN].nonzero()))

    all_genes = np.unique(all_genes)
    # there should just be one embedding, so we don't need to to some
    # kind of storage yoga
    self.data.obsm['embedding'] = self.data[:, all_genes].X.copy()


        
```

## Saving the results
Here, we just save the embedding as a data frame. 

```
  def _write_results(self):
      """
      Dummy write method. Probably useless as is.
      """
      try:
          file_path = op.join(self.data.uns['embedding_dir'], 'embedding.tsv')
          if scs.issparse(self.data.obsm['embedding']):
              saveme = self.data.obsm['embedding'].toarray()
          else:
              saveme = self.data.obsm['embedding']
          saveme = pd.DataFrame(saveme)
          saveme.index = self.data.obs.index
          saveme.to_csv(file_path, sep = '\t', index=True)

      except KeyError:
          print('Results not initialized.')
```

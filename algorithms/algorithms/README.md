# AlgorithmsWrapper

The AlgorithmsWrapper is the entry point for the algorithms. Since it may be required to implement a custom convergence criterion, or flexible combination of different algorithm components, it is possible to create a derived class from AlgorithmsWrapper. 

In many cases, if the procedure requires only iterative chaining of the tree components, it may not be necessary to implement a custom AlgorithmWrapper. In this case, the Algorithm can be parameterized with the required parameters.

```python
from algorithms.utils.data_utils import create_anndata_from_prefixes

#### REPLACE WITH ACTUAL DATA
filename = '/home/anne/Documents/netmap/data/tox-cd8'
output = '/home/anne/Documents/netmap/temp-res/initial_results'
prefix = ['GSM3568585_scRNA_D4.5_P14_Arm_1_']

# This is convienience code stacking some ```.mtx``` files.
data = create_anndata_from_prefixes(data_directory = filename, prefix=prefix)
####

# instantiate the AlgorithmWrapper with the BASIC Strategies defined in the other tutorials.
max_iterations = 1
n_clusters = 12
my_algorithm = AlgorithmWrapper(data,
                                cell_embedding_strategy=CellEmbeddingStrategy.BASIC,
                                clustering_strategy=ClusteringUpdateStategy.BASIC,
                                grn_inference_strategy=GRNInferrenceStrategy.BASIC,
                                initialization_strategy=InitializationStrategy.BASIC,
                                max_iterations= max_iterations,
                                output_directory = output_directory,
                                n_clusters = n_clusters
                                )
# Dummy parameters for tolerances                                
my_algorithm.run(GRN_convergence_tolerance=1, cluster_convergence_tolerance=1)
```

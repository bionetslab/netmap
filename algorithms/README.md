# Installation instructions

## Environment
### Conda

Some people are using conda/mamba as a package resolver, we assume you have conda installed. In the base environment intall mamba, the faster package resolver.
```
conda install mamba -n base -c conda-forge
```

The install the dependencies using the following command
```
mamba env create -f ../netmap.yml
conda activate netmap
```


## Repository structure
The repository is structured in several subfolders according to the step in the algorithm:

- [algorithms](./algorithms) Every algorithm configuration follows the same schema: Initialization, Iterative identification of the best cell partition based on GRNs, Finalization (save results, etc.) The AlgorithmsWrapper allows the developer to configure this process, by either chosing from existing stategies, or by implementing new components. Since it may be necessary to modify convergence criteria, or other components, the AlgorithmWrapper class can be extended, or modified to customize the algorithm.

We expect different strategies to be implemented in the following repositories. To ensure that the components can be combined with better flexibility, there is a FactoryClass which takes care of the instantiation of the Strategy, and an Abstract class which needs to be extended to implement the strategy. 

- [clustering](./clustering) 
- [embedding](./embedding)
- [inference](./inference)
- [initializers](./initializers) 

- [utils](./utils) Contains some utilities that are expected to be shared by all implementations, such as file saving etc. to enforce naming conventions etc.

## Implementing a custom Strategy. (e.g. A BasicClusteringStrategy)

If a developer wants to implement a custom Strategy, the following steps are required:
1. Create a derived class from the Abstract class and implement the abstract methods. For example, we can add a BasicClusteringStrategy by implementing the ClusteringUpdateWrapper.

**clustering/BasicClustering.py**
```python
from algorithms.clustering.AbstractClusteringUpdate import ClusteringUpdateWrapper

class BasicClustering(ClusteringUpdateWrapper):
    def __init__(self) -> None:
        super().__init__()
        pass
```

2. Add the Strategy name to the Stategy.py Enum. There is one enum per Step. E.g. We add the BASIC Stategy to the ClusteringUpdateStategy Enum.

**Strategy.py**
```python
class ClusteringUpdateStategy(Enum):
    BASIC  = 1    
```

3. Add the strategy to the Factory class so that the Strategy object can be intiantiated. Eg. We add possibility to instantiate BasicClustering() in the ClusteringFactory

**clustering/ClusteringFactory.py**
```python
class ClusteringFactory:
    
    def create_clustering_wrapper(self, type: ClusteringUpdateStategy):
        if type == ClusteringUpdateStategy.BASIC:
            return BasicClustering()     
```

4. The Clustering strategy can then be used by passing it to the AlgorithmWrapper instance, or a derived class.

**algorithms/AlgorithmWrapper.py**
```python
class AlgorithmWrapper(ABC):
    def __init__(self, 
                  data,
                  cell_embedding_strategy: CellEmbeddingStrategy, 
                  clustering_strategy: ClusteringUpdateStategy, 
                  grn_inference_strategy: GRNInferrenceStrategy,
                  initialization_strategy: InitializationStrategy,
                  max_iterations = 100) -> None:
        
        
        self.data = data
        self.max_iterations = max_iterations

        # save the variables
        self.cell_embedding_strategy = cell_embedding_strategy
        self.clustering_strategy = clustering_strategy
        self.grn_inference_strategy = grn_inference_strategy

        # initialize 
        self.GRN_inferrence = GRNInferenceFactory().create_inference_wrapper(type=grn_inference_strategy)
        self.cell_embedding = EmbeddingFactory().create_embedding_wrapper(type=cell_embedding_strategy)
        self.clustering = ClusteringFactory().create_clustering_wrapper(type=clustering_strategy)

        self.initializer = InitializerFactory().create_initializer_wrapper(type=initialization_strategy)
    
```

5. Finally, the new Strategy can be used in the TestRunner class. (Here, we assume to have implemented a BasicStategy for all Steps in the algorithm)

**TestRunner.py**
```
def run_tests():
    ###
    data = np.random.random((2000,39))
    data = pd.DataFrame(data)

    my_algorithm = AlgorithmWrapper(data,
                                    cell_embedding_strategy=CellEmbeddingStrategy.BASIC,
                                    clustering_strategy=ClusteringUpdateStategy.BASIC,
                                    grn_inference_strategy=GRNInferrenceStrategy.BASIC,
                                    initialization_strategy=InitializationStrategy.BASIC,
                                    max_iterations= 12
                                    )
```

That's it!




## Abstract classes
- InitializationWrapper; input: cell list, initial clustering or number of clusters k; output: cell clustering with k clusters
- GRNInferenceWrapper; input: expression data, cell clustering, old GRNs; output: new GRNs (one for each of the k clusters)
- CellEmbeddingWrapper; input: expression data, cluster-specific GRNs; output: low-dimensional embedding representing the GRNs (easy solution: expression of genes contained in GRNs, better solution: expressive power of GRN i yields z_i(c))
- ClusteringUpdateWrapper: input: GRN-based cell embedding, old clustering; output: new cell clustering with k clusters
- AlgorithmWrapper; meta object holding the data, Initialization wrapper, GRNInferenceWrapper, CellEmbeddingWrapper, ClusteringUpdateWrapper

## Basic implementations:
- IntializationWrapper: either user-defined cell clustering or random split into k clusters with user-provided k
- GRNInferenceWrapper: for each cluster i, infer GRN using any fast GRN inference method (e.g., ACRANCNeAP, https://github.com/bionetslab/grn-confounders/blob/main/src/confinspect/ARACNEWrapper.py, 10.1093/bioinformatics/btw216). Then, define cluster-specific GRN i as edge set of inferrred GRN i - union of edge sets of all other GRNs. Or else, just keep the top m edges of each inferred GRN i. Enforce same number of nodes/genes for each GRN. Check for GRN-wise convergence (edge-based).
- CellEmbeddingWrapper: for each cell, use only expression of genes contained in the k GRNs as low-dimenensional representation 
- ClusteringUpdateWrapper: Use spectral clustering (or any other clustering appraoch) to compute new clusters based on low-dimensional representation. Check for cluster-wise convergence.

## Termination criteria
- Cluster-wise and/or GRN-wise convergence.
- Number of iterations.
- Time limit.
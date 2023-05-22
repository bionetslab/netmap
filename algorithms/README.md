# Algorithms

## Repository structure
The repository is structured in several subfolders according to the step in the algorithm:

### [algorithms](./algorithms) 
Every algorithm configuration follows the same schema: 
1. Initialization: the initial cell label assignment, e.g. random, or based on marker genes
2. Iterative identification of the best cell partition based on GRNs
    - Identification of GRNs for each cell cluster
    - Computation of Embedding based on the GRN
    - COmputation of new labels for the cell using the new embedding
3. Finalization (save results, etc.) 

The AlgorithmsWrapper allows the developer to configure this process, by either chosing from existing stategies, or by implementing new components. Since it may be necessary to modify convergence criteria, or other components, the AlgorithmWrapper class can be extended, or modified to customize the algorithm. We expect different strategies to be implemented in the following repositories. To ensure that the components can be combined with better flexibility, there is a FactoryClass which takes care of the instantiation of the Strategy, and an Abstract class which needs to be extended to implement the strategy. 

### [clustering](./clustering) 
### [embedding](./embedding)
### [inference](./inference)
### [initializers](./initializers) 
### [utils](./utils) 

Contains some utilities that are expected to be shared by all implementations, such as file saving etc. to enforce naming conventions etc.


## Data structure
Generally, all data should be passed along as a single Anndata object. The classes do not depend on any other variables. Parameters can be passed via the ```AnnData.uns``` field. This has the advantage, that the state of the object is preserved upon saving the Anndata object and the disadvantage that developers need to adhere to certain standards/ be careful when implemeting the classes, to be sure to use the correct fields in the Anndata object.

The format for AnnData is described in the documentation of the [package](https://anndata.readthedocs.io/en/latest/index.html). Generally speaking AnnData allows us to store the data, column and row annotations in ```.obs`` and ```.var```, row-, and column-aligned matrix shaped data ```.obsm``` and ```.obsp``` as well as row and column aligned graph data ```.opsp``` and ```.varp```. Unstructured data can be stored as a dictionary in ```.uns``.  

For this project, that means that Gene Regulatory Networks (GRNs) are stores in ```.varp``` as they represent feature associated graph data, the embeddings will be saved in ```obsm``` as they represent observation associated matrix shaped data, and the clustering labels will be saved in ```.obs``` as they represent vector shaped annotations of the observations. (More details on this will be added in the subdirectories for each stage of the algorithm.)


## Implementing a custom Strategy -- top level view (e.g. A BasicClusteringStrategy)
If a developer wants to implement a custom Strategy, for example a Custom Clustering Strategy, they need to follow the steps below. Details on how to implement the custom strategies for each step can be found in the respective subfolders.

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


6. We recommend adding a configuration file instead of parsing the parameters via the command line, to increase reproducibility and readibility. We use ```.yaml``` files to define the parameters. The utils folder contains a basic yaml parser.

```python
strategy:
  InitializationStrategy: BASIC
  GRNInferrenceStrategy: BASIC
  ClusteringUpdateStategy: BASIC 
  CellEmbeddingStrategy: BASIC
algorithm:
  max_iterations: 1
  clustering_tolerance: 0.75
  grn_consistency: 0.75
input:
  directory: '/home/anne/Documents/netmap/data/tox-cd8'
  filename: 
  prefix: ['GSM3568585_scRNA_D4.5_P14_Arm_1_']
output:
  directory: /home/anne/Documents/netmap/temp-res/initial_results
```
7. Lastly, when implementing a new strategy, the developer will add documentation on the newly implemented module in the newly implemented class, as well as a documentation file for the overall stategy, in the documentation folder. This includes a description of the strategies for Initialization, GRN Inferrence, Embedding, and Clustering. Users should be able to run the strategy based on the documentation, including the installation and execution of external tools. The documentation should link the the configuration file. See [BASIC.md](../documentation/BASIC.md) as an example.

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

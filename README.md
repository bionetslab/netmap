# Netmap -- Explainable dimensionality reduction.

This repository implements an iterative procedure to infer small explanatory GRNs to be used as an axis label in our novel dimensionality reduction strategy for HTP molecular measurements. The algorithm follows an EM-strategy, where an initial cell type assignment is chosen, GRNs are inferred given the cell type assignment, the explanatory power of the GRNs for the given clustering is determined. The algorithm terminates when it reaches convergence, i.e. when the cell type assignment and the GRNs remain stable (within a tolerance).

## The strategy.
1. Initial cell type assigmnent (e.g. random or user defined)
2. (Differential) GRN inferrence for each cluster.
3. Test of explanatory power  (How??), new cell type assigment.
4. Test of convergence.

## Running the tool

### Installation
### Parameters
### Results


## The repository
The [algorithms](./algorithms) folder contains implementations of different strategies to simultanously infer cell type assignment and GRNs. For information on how to add new stategies or implementation details, please refer to this repository.

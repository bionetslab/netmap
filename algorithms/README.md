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



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
The repository is structured in several subfolders as shown below. Every folder contains a separate README that explains the contents of this folder in greater detail. Here, some general information will be provided. The algorithms is of interest for developers who want to implement their own strategy.

```
.
├── algorithms
│   ├── algorithms
│   ├── clustering
│   ├── embedding
│   ├── inference
│   ├── initializers
│   ├── __init__.py
│   ├── __pycache__
│   ├── README.md
│   ├── run.py
│   ├── Strategy.py
│   ├── TestRunner.py
│   └── utils
├── external
│   ├── ARACNe-AP
│   └── README.md
├── LICENSE
├── netmap.yml
└── README.md

```

## Documentation 
The documentation is created using Sphinx and hosted on the ...

## Using the code

Our code is available as ...



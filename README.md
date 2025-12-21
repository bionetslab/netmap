# NETMAP (single-cell gene regulatory networks using XAI)

## Installation
The repository is installable using pixi and pip. If you want to try out netmap, you can clone the repository, and install a kernel via pixi.
Note: We do not include torch and cupy. These need to be installed separately when using pip, and will be installed automatically when using pixi installation

### Pixi installation
1. Download and setup [pixi.sh](https://pixi.prefix.dev/latest/).
2. Install the jupyer kernel which includes torch and cupy and netmap. You can then set pixi_netmap as a kernel in your jupyterlab, or vscode environment
3. You can add extra packages using ``` pixi add my-package ``` and reinstall the kernel to add extra packages.
```{bash}
git clone https://github.com/bionetslab/netmap
cd netmap
pixi run install-kernel
```

### pip install
A local, editable install in a conda environment can be done as follows. You need to install extra denpendencies yourself e.g. torch, cupy depending on you system.

```{bash}
git clone https://github.com/bionetslab/netmap
cd netmap
conda create -n netmap_env
conda activate netmap_env
pip install -e .
```

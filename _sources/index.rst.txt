NETMAP Documentation
====================

NETMAP infers single-cell gene regulatory networks (GRNs) using explainable AI.
An ensemble of count-distribution autoencoders is trained on scRNA-seq data,
then Captum attribution methods score directed gene–gene influences per cell.
The full attribution matrix (cells × genes²) is stored as an AnnData object.

Pipeline overview
-----------------

1. **Model training** — :mod:`netmap.model` trains an ensemble of
   :class:`~netmap.model.zinbautoencoder.ZINBAutoencoder` or
   :class:`~netmap.model.nbautoencoder.NegativeBinomialAutoencoder` models
   with early stopping via :func:`~netmap.model.train_model.create_model_zoo`.

2. **GRN inference** — :func:`~netmap.grn.inferrence.inferrence` applies
   Captum attribution (GradientShap, GuidedBackprop, or Deconvolution) across
   all model–target combinations and writes Parquet-backed output.

3. **Masking** — :mod:`netmap.masking` annotates edges with co-expression
   support (:mod:`~netmap.masking.internal`) and external reference GRN
   overlap (:mod:`~netmap.masking.external`).

4. **Downstream** — :mod:`netmap.downstream` clusters the GRN, ranks
   edges, builds signed regulons, and produces interactive network
   visualisations.

Installation
------------

.. code-block:: bash

   # Recommended — includes GPU deps (PyTorch, CuPy), CUDA 12.8, linux-64 only
   pixi run install-kernel

   # pip — GPU dependencies must be added manually
   pip install -e .

.. toctree::
   :maxdepth: 4
   :caption: API Reference

   api

"""YAML-serializable configuration dataclass for the NETMAP pipeline."""

from dataclasses import dataclass, field
from typing import List
import yaml



@dataclass
class NetmapConfig:
    """Configuration dataclass for the NETMAP pipeline.

    Holds all parameters required to run the NETMAP pipeline stages (model
    training, GRN inference, masking, and downstream analysis).  Instances can
    be serialised to / deserialised from YAML via :meth:`write_yaml` and
    :meth:`read_yaml`.

    Fields (all keyword-argument constructors with defaults):

    - ``input_data`` (str) — path to the input AnnData ``.h5ad`` file.
      Default ``"data.h5ad"``.
    - ``layer`` (str) — AnnData layer used as model input; ``'X'`` for the
      main expression matrix. Default ``'X'``.
    - ``output_directory`` (str) — root directory for pipeline outputs.
      Default ``"netmap"``.
    - ``transcription_factors`` (str) — path to a one-TF-per-line file.
      Default ``""``.
    - ``tf_only`` (bool) — restrict sources to TFs when ``True``.
      Default ``True``.
    - ``penalize_error`` (bool) — include reconstruction-error penalty.
      Default ``True``.
    - ``adata_filename`` (str) — filename for the saved GRN AnnData inside
      ``output_directory``. Default ``"grn_lrp.h5ad"``.
    - ``grn`` (str) — filename for the exported GRN TSV. Default ``"grn_lrp.tsv"``.
    - ``masking_percentage`` (float) — fraction of cells for masking.
      Default ``0.1``.
    - ``print_every`` (int) — log training every N epochs. Default ``100``.
    - ``optimizer`` (str) — PyTorch optimiser name. Default ``'Adam'``.
    - ``learning_rate`` (float) — optimiser learning rate. Default ``0.005``.
    - ``epochs`` (int) — maximum training epochs per model. Default ``10000``.
    - ``n_models`` (int) — ensemble size. Default ``20``.
    - ``validation_size`` (float) — early-stopping validation fraction.
      Default ``0.2``.
    - ``model`` (str) — autoencoder class name
      (``'NegativeBinomialAutoencoder'`` or ``'ZINBAutoencoder'``).
      Default ``"NegativeBinomialAutoencoder"``.
    - ``xai_method`` (str) — Captum attribution method
      (``'GradientShap'``, ``'GuidedBackprop'``, ``'Deconvolution'``).
      Default ``"GradientShap"``.
    - ``aggregation_strategy`` (str) — attribution aggregation strategy.
      Default ``'mean'``.
    """

    input_data: str =  "data.h5ad"
    layer: str = 'X'
    output_directory: str =  "netmap"
    transcription_factors: str =  ""
    tf_only: bool = True
    penalize_error: bool = True
    adata_filename: str =  "grn_lrp.h5ad"
    grn: str = "grn_lrp.tsv"
    masking_percentage: float = 0.1
    print_every: int = 100
    optimizer: str = 'Adam'
    learning_rate: float = 0.005
    epochs: int = 10000
    n_models: int = 20
    validation_size: float  = 0.2
    model:str =  "NegativeBinomialAutoencoder"
    xai_method:str =  "GradientShap"
    aggregation_strategy:str = 'mean'



    @classmethod
    def read_yaml(cls, yaml_file):
        """Construct a :class:`NetmapConfig` from a YAML file.

        Reads the YAML file at ``yaml_file``, prints the parsed dictionary for
        debugging, and returns a new :class:`NetmapConfig` instance populated
        with the values found in the file.  Keys in the YAML must match the
        field names of :class:`NetmapConfig` exactly.

        Args:
            yaml_file (str): Path to the YAML configuration file to read.

        Returns:
            NetmapConfig: A new instance initialised from the YAML contents.
        """
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            print(data)
        return cls(**data)

    def write_yaml(self, yaml_file):
        """Serialise this configuration to a YAML file.

        Writes all fields of the current instance to ``yaml_file`` in YAML
        format, preserving field order (``sort_keys=False``).  The resulting
        file can be loaded back with :meth:`read_yaml`.

        Args:
            yaml_file (str): Destination path for the YAML file.  The file is
                created or overwritten if it already exists.

        Returns:
            None
        """
        with open(yaml_file, 'w') as f:
            yaml.dump(self.__dict__, f, sort_keys=False)

from dataclasses import dataclass, field
from typing import List
import yaml



@dataclass
class NetmapConfig:
    input_data: str =  "data.h5ad"
    layer: str = 'X'
    output_directory: str =  "netmap"
    transcription_factors: str =  "/data_nfs/datasets/SCENIC_DB/tf_lists/allTFs_hg38.txt"
    tf_only: bool = True
    penalize_error: bool = True
    adata_filename: str =  "grn_lrp.h5ad"
    grn: str = "grn_lrp.tsv"
    masking_percentage: float = 0.3
    masking_value: int = 0
    print_every: int = 100
    optimizer: str = 'Adam'
    learning_rate: float = 0.005
    epochs: int = 150
    overwrite: bool = True
    n_models: int = 2
    n_top_edges: int =  100
    test_size: float  = 0.3
    edge_count:int = 10000
    model:str =  "standard"
    loss_fn:str =  "LogCoshLoss"
    xai_method:str =  "GradientShap"
    top_perc:bool =  False
    percentile:int =  10
    penalty:str =  'pingouin.pcor'
    raw_attribution:bool = False
    aggregation_strategy:str = 'mean'
    use_differential:bool = True



    @classmethod
    def read_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            print(data)
        return cls(**data)
    
    def write_yaml(self, yaml_file):
        with open(yaml_file, 'w') as f:
            yaml.dump(self.__dict__, f, sort_keys=False)
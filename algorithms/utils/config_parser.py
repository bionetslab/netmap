import yaml
from algorithms.utils.expceptions import ConfigValidationError
import pandas as pd

def parse_configuration_file(yml_config):

    """
    Parse configuration file, ensure that strategy has been configured. Returns the configuration
    as a flattened dictionary.

    :y
    """

    with open(yml_config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Check if every Strategy has a values
    if len(set(['InitializationStrategy', 'GRNInferrenceStrategy', 'ClusteringUpdateStategy', 'CellEmbeddingStrategy']).intersection(set(config['strategy'].keys())))<4:
        raise ConfigValidationError('Missing mandatory configuration options')
    
    configuration = pd.json_normalize(config)
    configuration_dict = {}
    for col in configuration.columns:
        configuration_dict[col] = configuration.loc[0,col]
    configuration_dict

    return configuration_dict

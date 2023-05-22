import yaml
from algorithms.utils.expceptions import ConfigValidationError


def parse_configuration_file(yml_config):

    with open(yml_config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Check if every Strategy has a values
    if len(set(['InitializationStrategy', 'GRNInferrenceStrategy', 'ClusteringUpdateStategy', 'CellEmbeddingStrategy']).intersection(set(config['strategy'].keys())))<4:
        raise ConfigValidationError('Missing mandatory configuration options')
    
    return config
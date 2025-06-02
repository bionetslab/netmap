
import yaml

def write_config(c, file):
    with open(file, "w") as handle:
        yaml.safe_dump(c, handle)
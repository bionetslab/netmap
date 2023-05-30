import decoupler as dc
import pandas as pd
import os.path as op


if __name__ == '__main__':
    dorothea_path = op.join('data/dorothea.tsv')
    net = dc.get_dorothea(organism='mouse', levels=['A'])
    net.to_csv(dorothea_path, sep='\t', index= False)
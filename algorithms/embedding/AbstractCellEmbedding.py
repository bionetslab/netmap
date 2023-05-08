from abc import ABC, abstractmethod
import os.path as op
import pandas as pd
import numpy as np
import scipy as sc


class CellEmbeddingWrapper(ABC):
    def __init__(self, data) -> None:
        self.previous_labels = {}
        self.current_labels = {}
        self.data = data


    
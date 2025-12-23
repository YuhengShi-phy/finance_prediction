import os
from typing import List
import pandas as pd
import numpy as np


class Predictor:
    def __init__(self):
        pass

    def predict(self, data: List[pd.DataFrame]) -> List[List[int]]:
        pass

    def load_model(self, model_path: str):
        pass

    def preprocess(self, data: List[pd.DataFrame]):
        pass

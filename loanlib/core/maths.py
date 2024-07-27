import pandas as pd
from typing import Tuple, List

class Curve:

    def __init__(self, df: pd.DataFrame, pivots: Tuple[str] = ()):
        self.df = df
        self.pivots = pivots


def SMM(df: pd.DataFrame):
    pass

def MDR(df: pd.DataFrame):
    pass

def CPR(df: pd.DataFrame, pivots: Tuple[str] = ()):
    seasoning = df['seasoning']

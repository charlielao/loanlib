import pandas as pd
from typing import Tuple, List, Callable

class CurveBuilder:
    def __init__(self, df: pd.DataFrame, index: str = 'seasoning', pivots: List[str] = ()):
        self.df = df

    @property
    def curve(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def plot(self):
        import matplotlib.pyplot as plt
        return plt.plot(self.curve)


def generate_base_metric( metric_name: str, metric_calc_func: Callable, df: pd.DataFrame, index: str = 'seasoning', pivots: List[str] = () ):
    transformed_df = df.reset_index().set_index([index])
    transformed_df[metric_name] = transformed_df.apply(axis=1, func=metric_calc_func)
    if not pivots:
        return transformed_df.groupby(index)[metric_name].sum()
    else:
        return pd.pivot_table(transformed_df, values = metric_name, index = index, columns = pivots, aggfunc = 'sum')


def smm(df: pd.DataFrame, index: str = 'seasoning', pivots: List[str] = ()):
    func = lambda row: 0.0 if not row['prepaid_in_month'] else (row['Payment Made'] - row['Payment Due']) / (row['current_balance'])
    return generate_base_metric( 'SMM', func, df, index, pivots )


def mdr(df: pd.DataFrame, index: str = 'seasoning', pivots: List[str] = ()):
    func = lambda row: 0.0 if not row['prepaid_in_month'] else (row['Payment Due'] - row['Payment Made']) / (row['current_balance'])
    return generate_base_metric( 'MDR', func, df, index, pivots )


def cpr(df: pd.DataFrame, index: str = 'seasoning', pivots: List[str] = ()):
    cpr = smm(df, index, pivots)
    cpr['CPR'] = cpr['SMM'].apply(lambda x: (1 - x) ^ 12)
    return cpr


def cdr(df: pd.DataFrame, index: str = 'seasoning', pivots: List[str] = ()):
    cdr = mdr(df, index, pivots)
    cdr['CDR'] = cdr['MDR'].apply(lambda x: (1 - x) ^ 12)
    return cdr


def recover_curve(df: pd.DataFrame, index: str = 'seasoning', pivots: List[str] = ()):
    func = lambda row: row['recovery_percent']
    return generate_base_metric( 'Recovery', func, df, index, pivots )


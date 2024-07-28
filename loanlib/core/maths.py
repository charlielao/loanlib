import pandas as pd
from typing import Tuple, List, Callable


class CurveBuilder:
    def __init__(self, df: pd.DataFrame, index: str = 'seasoning', pivots: List[str] = ()):
        self.df = df
        self.index = index
        self.pivots = pivots

    def curve(self, curve_type : str) -> pd.DataFrame:
        args = (self.df, self.index, self.pivots)
        match curve_type:
            case 'SMM':
                return self.smm(*args)
            case 'MDR':
                return self.mdr(*args)
            case 'CPR':
                return self.cpr(*args)
            case 'CDR':
                return self.cdr(*args)
            case 'Recovery':
                return self.recover_curve(*args)
            case _:
                raise NotImplementedError(f'{curve_type} is not implemented')

    def plot(self, curve_type: str):
        import matplotlib.pyplot as plt
        return plt.plot(self.curve(curve_type))

    @classmethod
    def _generate_base_metric(cls, metric_name: str, metric_calc_func: Callable, df: pd.DataFrame, index: str, pivots: List[str]):
        transformed_df = df.reset_index().set_index([index])
        transformed_df[metric_name] = transformed_df.apply(axis=1, func=metric_calc_func)
        if not pivots:
            return transformed_df.groupby(index)[metric_name].sum()
        else:
            return pd.pivot_table(transformed_df, values = metric_name, index = index, columns = pivots, aggfunc = 'sum')

    @classmethod
    def smm(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        func = lambda row: 0.0 if not row['prepaid_in_month'] else (row['Payment Made'] - row['Payment Due']) / (row['current_balance'])
        return cls._generate_base_metric( 'SMM', func, df, index, pivots )

    @classmethod
    def mdr(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        func = lambda row: 0.0 if not row['prepaid_in_month'] else (row['Payment Due'] - row['Payment Made']) / (row['current_balance'])
        return cls._generate_base_metric( 'MDR', func, df, index, pivots )

    @classmethod
    def cpr(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        cpr = cls.smm(df, index, pivots)
        cpr['CPR'] = cpr['SMM'].apply(lambda x: (1 - x) ^ 12)
        return cpr

    @classmethod
    def cdr(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        cdr = cls.mdr(df, index, pivots)
        cdr['CDR'] = cdr['MDR'].apply(lambda x: (1 - x) ^ 12)
        return cdr

    @classmethod
    def recover_curve(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        func = lambda row: row['recovery_percent']
        return cls._generate_base_metric( 'Recovery', func, df, index, pivots )


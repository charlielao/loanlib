import pandas as pd
from typing import Tuple, List, Callable


class LoanMetrics:
    def __init__(self, df: pd.DataFrame|str, index: str = 'seasoning', pivots: List[str] = ()):
        if isinstance(df, str):
            from loanlib.data_loader import DataLoader
            loader = DataLoader(df)
            df = loader.create_features(loader.combined_data_frame)
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
                raise NotImplementedError(f'{curve_type} Curve is not implemented')

    def plot(self, curve_type: str):
        if self.pivots:
            raise ValueError('Cannot plot pivoted data')
        import matplotlib.pyplot as plt
        return plt.plot(self.curve(curve_type)[curve_type.lower()])

    @classmethod
    def _generate_base_metric(cls, metric_name: str, df: pd.DataFrame, index: str, pivots: List[str]):
        transformed_df = df.reset_index().set_index([index])
        if not pivots:
            resulting_df = transformed_df.groupby(index)[metric_name].mean().reset_index()
            resulting_df.set_index(index, inplace=True)
            return resulting_df
        else:
            return pd.pivot_table(transformed_df, values=metric_name, index=index, columns=pivots, aggfunc='mean')

    @classmethod
    def smm(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        return cls._generate_base_metric( 'smm', df, index, pivots )

    @classmethod
    def mdr(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        return cls._generate_base_metric( 'mdr', df, index, pivots )

    @staticmethod
    def annualize(x):
        return 1.0 - (1.0 - x) ** 12

    @classmethod
    def cpr(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        cpr = cls.smm(df, index, pivots)
        if pivots:
            return cpr.apply(LoanMetrics.annualize)
        else:
            cpr['cpr'] = cpr['smm'].apply(LoanMetrics.annualize)
            cpr.drop(columns=['smm'], inplace=True)
            return cpr

    @classmethod
    def cdr(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        cdr = cls.mdr(df, index, pivots)
        if pivots:
            return cdr.apply(LoanMetrics.annualize)
        else:
            cdr['cdr'] = cdr['mdr'].apply(LoanMetrics.annualize)
            cdr.drop(columns=['mdr'], inplace=True)
            return cdr

    @classmethod
    def recover_curve(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        return cls._generate_base_metric('recovery', df, index, pivots)


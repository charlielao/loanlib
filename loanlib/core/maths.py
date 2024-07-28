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
    def _generate_base_metric(cls, metric_name: str, df: pd.DataFrame, index: str, pivots: List[str]):
        transformed_df = df.reset_index().set_index([index])
        if not pivots:
            resulting_df = transformed_df.groupby(index, as_index=False)[metric_name].sum()
            resulting_df.index.name = index
            return resulting_df
        else:
            return pd.pivot_table(transformed_df, values=metric_name, index=index, columns=pivots, aggfunc = 'sum')

    @classmethod
    def smm(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        return cls._generate_base_metric( 'smm', df, index, pivots )

    @classmethod
    def mdr(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        return cls._generate_base_metric( 'mdr', df, index, pivots )

    @classmethod
    def cpr(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        cpr = cls.smm(df, index, pivots)
        cpr['cpr'] = cpr['smm'].apply(lambda x: 1.0 - (1.0 - x) ** 12)
        return cpr

    @classmethod
    def cdr(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        cdr = cls.mdr(df, index, pivots)
        cdr['cdr'] = cdr['mdr'].apply(lambda x: 1.0 - (1 - x) ** 12)
        return cdr

    @classmethod
    def recover_curve(cls, df: pd.DataFrame, index: str, pivots: List[str]):
        return cls._generate_base_metric('recovery', df, index, pivots )


import pandas as pd
import datetime
from typing import List


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.all_spreadsheets = pd.read_excel(self.file_path, sheet_name=None)
        self.data_sheet_names = tuple(sheet_name for sheet_name in self.all_spreadsheets if self._sheet_contains_data(sheet_name))
        self.data_frames = {sheet_name: self._format_data_frame(self.all_spreadsheets[sheet_name]) for sheet_name in self.data_sheet_names}
        self.combined_data_frame = self._combine_data_frames()

    @classmethod
    def _sheet_contains_data(cls, sheet_name: str):
        return sheet_name.startswith('DATA')

    @classmethod
    def _data_frame_is_static(cls, df: pd.DataFrame):
        return not cls._data_frame_is_time_series(df)

    @classmethod
    def _data_frame_is_time_series(cls, df: pd.DataFrame):
        all_cols = df.columns
        index_in_column = isinstance(all_cols[0], str) and all_cols[0].lower() == 'loan_id'
        return all((isinstance(col, (datetime.datetime, datetime.date)) for col in all_cols[1 if index_in_column else 0:]))

    @classmethod
    def _validate_data_frame(cls, df: pd.DataFrame):
        'should implement this to make sure data is clean e.g. only month end dates, no overlap etc.'
        pass

    @classmethod
    def _format_data_frame(cls, df: pd.DataFrame):
        if cls._data_frame_is_static(df):
            df.columns = df.iloc[1]
            df = df.iloc[2:, 1:]
            df.set_index('loan_id', inplace=True)
            df = df.apply(lambda x: x if not isinstance(x, datetime.datetime) else x.date(), axis=1)
        else:
            df.columns = ('loan_id', *tuple(timestamp.date() for timestamp in df.columns[1:]))
            df.set_index('loan_id', inplace=True)
            df.fillna(value=0.0,inplace=True)
        df.index.names = ['ID']
        return df

    @staticmethod
    def _melt_time_series(df: pd.DataFrame, source_name: str):
        return df.reset_index().melt(id_vars='ID', var_name='Date', value_name=source_name)

    def _combine_data_frames(self):
        from functools import reduce
        time_series_dfs = []
        static_dfs = []
        for df_name, df in self.data_frames.items():
            if self._data_frame_is_time_series(df):
                time_series_dfs.append(self._melt_time_series(df, df_name.replace('DATA-', '')))
            else:
                static_dfs.append(df)
        if len(static_dfs) != 1:
            raise ValueError('There should be exactly one static DataFrame in the data_frames attribute')
        static_df = static_dfs[0]
        ts_df_combined = reduce(lambda left, right: pd.merge(left, right, on=['ID', 'Date'], how='outer'), time_series_dfs)
        dates = ts_df_combined['Date'].unique()
        index = pd.MultiIndex.from_product([static_df.index, dates], names=['ID', 'Date'])
        static_repeated = static_df.reindex(index, level=0).reset_index()
        df_final = pd.merge(ts_df_combined, static_repeated, on=['ID', 'Date'])
        df_final = df_final.set_index(['ID', 'Date'], drop=False)
        return df_final


def augment_data_frame(df: pd.DataFrame, skipped_features: List[str] = []):
    from loanlib.core import custom_feature
    from graphlib import TopologicalSorter
    import inspect
    register = custom_feature.custom_column_register
    function_name_map = dict(inspect.getmembers(custom_feature, inspect.isfunction))
    all_custom_funcs = set(register.keys())
    dependency_graph = {func: {dep for dep in deps if dep in all_custom_funcs } for func, deps in register.items()}
    for func_name in TopologicalSorter(dependency_graph).static_order():
        if func_name not in skipped_features:
            func = function_name_map[func_name]
            has_dependencies = len( register[func_name] ) > 0
            df[func_name] = pd.Series(df.groupby(level=['ID']).apply(func).values.flatten()
                                      if has_dependencies else func(df)).values
    df.drop(columns=df.index.names, inplace=True)
    return df


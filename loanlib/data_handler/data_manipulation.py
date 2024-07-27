import pandas as pd

def augment_data_frame(df: pd.DataFrame):
    from loanlib.core import custom_column_func
    from graphlib import TopologicalSorter
    import inspect
    register = custom_column_func.custom_column_register
    function_name_map = dict(inspect.getmembers(custom_column_func, inspect.isfunction))
    all_custom_funcs = set(register.keys())
    dependency_graph = {func: {dep for dep in deps if dep in all_custom_funcs } for func, deps in register.items()}
    for func_name in TopologicalSorter(dependency_graph).static_order():
        func = function_name_map[func_name]
        has_dependencies = len( register[func_name] ) > 0
        df[func_name] = pd.Series(df.groupby(level=['ID']).apply(func).values.flatten()
                                  if has_dependencies else func(df)).values

    return df


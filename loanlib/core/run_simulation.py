import numpy as np
from multiprocessing import Pool
import pandas as pd
from loanlib.core.mortgage_cashflow_model import MortgageCashflowModel


BASE_CONFIG = {
    'month_post_reversion': -22,
    'seasoning': 2,
    'current_balance': 100_000,
    'fixed_pre_reversion_rate': 3.94/100.0,
    'post_reversion_margin': 4.94/100.0,
    'month_to_maturity': 178,
    'repayment_method': 'Interest Only',
    'ls': 0.2,
    'recovery_lag': 6,
    'cpr': pd.DataFrame({'cpr': [0.02]*205, 'month_post_reversion': range(-24, 181)}).set_index('month_post_reversion')['cpr'],
    'cdr': pd.DataFrame({'cdr': [0.02]*205, 'month_post_reversion': range(-24, 181)}).set_index('month_post_reversion')['cdr'],
    'boe_base_rate': np.array([4.5/100.0]*200)

}

def run_simulation(config_override: dict):
    configuration = {**BASE_CONFIG, **config_override}
    model = MortgageCashflowModel(**configuration)
    model.run()
    return model.df


if __name__ == '__main__':
    import os
    configs = [{}, {}]
    # instantiating process with arguments
    with Pool(os.cpu_count()) as p:
        results = (p.map(run_simulation, configs))

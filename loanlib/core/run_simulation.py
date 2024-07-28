import inspect
import multiprocessing
from multiprocessing import Process
import numpy as np
import pandas as pd
from loanlib.core.mortgage_cashflow_model import MortgageCashflowModel

configuration = {
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
model = MortgageCashflowModel(**configuration)
model_outputs = {}
attrs = ((name, getattr(model, name)) for name in dir(model))
for name, attr in attrs:
    if inspect.ismethod(attr) and 'jitted' not in name and 'forecast_month' in inspect.signature(attr).parameters:
        try:
            print(name, attr(3))
        except TypeError as e:
            print(e)
            # Can't handle methods with required arguments.
            pass
if False:
    def print_func(continent='Asia'):
        print('The name of continent is : ', continent)

    if __name__ == "__main__":  # confirms that the code is under main function
        names = ['America', 'Europe', 'Africa']
        procs = []
        proc = Process(target=print_func)  # instantiating without any argument
        procs.append(proc)
        proc.start()

        # instantiating process with arguments
        for name in names:
            # print(name)
            proc = Process(target=print_func, args=(name,))
            procs.append(proc)
            proc.start()

        # complete the processes
        for proc in procs:
            proc.join()
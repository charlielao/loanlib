import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from functools import cached_property, lru_cache
from typing import List, Dict
from numba import njit

_INTEREST_ONLY_LITERAL = 'Interest Only'
_REPAYMENT_METHODS = {_INTEREST_ONLY_LITERAL}
'''
using numba instead because numpy cannot support recursive relationship
could potentially use a context manager to avoid passing forecast_month everytime

should use a dynamic programming table to build up the table, but determining the order is a bit involved needs topology sort
need to specify dependency

#can do Similar sort of computational graph so everything only has to be evaluated once automatically, or alternatively caching
#could potentially refactored out all the numpy computations in fact
'''


class MortgageCashflowModel:

    def __init__(self, month_post_reversion: int, seasoning: int, current_balance : float, fixed_pre_reversion_rate: float,
                    post_reversion_margin: float, month_to_maturity: int, repayment_method: str, ls: float, recovery_lag: int,
                    cpr: pd.DataFrame | float, cdr: pd.DataFrame | float, boe_base_rate: np.array):
        self.month_post_reversion = month_post_reversion
        self.seasoning = seasoning
        self.current_balance = current_balance
        self.fixed_pre_reversion_rate = fixed_pre_reversion_rate
        self.post_reversion_margin = post_reversion_margin
        self.month_to_maturity = month_to_maturity
        self.repayment_method = repayment_method
        self.ls = ls
        self.recovery_lag = recovery_lag
        self.boe_base_rate = boe_base_rate
        self.input_cpr = cpr if isinstance(cpr, pd.Series) else cpr['cpr'] if isinstance(cpr, pd.DataFrame) else pd.DataFrame({'cpr': [cpr]*205, 'month_post_reversion': range(-24, 181)}).set_index('month_post_reversion')['cpr']
        self.input_cdr = cdr if isinstance(cdr, pd.Series) else cdr['cdr'] if isinstance(cdr, pd.DataFrame) else pd.DataFrame({'cdr': [cdr]*205, 'month_post_reversion': range(-24, 181)}).set_index('month_post_reversion')['cdr']
        self.number_month_forecasted = len(self.boe_base_rate)
        self.forecast_months = np.arange(1, self.number_month_forecasted+1)
        self.verify_parameters()
        self.df = self.run()


    def verify_parameters(self):
        if self.fixed_pre_reversion_rate < 0.0 or self.fixed_pre_reversion_rate > 1.0:
            raise ValueError('Fixed pre-reversion rate should be between 0 and 1')

        if self.post_reversion_margin < 0.0 or self.post_reversion_margin > 1.0:
            raise ValueError('Post-reversion margin should be between 0 and 1')

        if self.repayment_method not in _REPAYMENT_METHODS:
            raise NotImplementedError('Only interest only repayment method is supported')

        #if self.input_cpr.index != self.input_cdr.index:
        #    raise ValueError('CPR and CDR indices should match')

    def run(self):
        import inspect
        df = {}
        all_attributes = frozenset(dir(self))
        relevant_attributes = []
        for name in all_attributes:
            attr = getattr(self, name)
            new_attr_name = name[1:]
            # needs better way to enforce ways to build rows, too brittle currently
            if inspect.ismethod(attr) and 'forecast_month' in inspect.signature(attr).parameters \
                    and 'jitted' not in name and name[0] == '_' and new_attr_name not in all_attributes:
                relevant_attributes.append((attr, new_attr_name))

        for forecast_month in self.forecast_months:
            for attr, attr_name in relevant_attributes:
                if attr_name not in df:
                    df[attr_name] = np.zeros(len(self.forecast_months))
                df[attr_name][forecast_month-1] = attr(forecast_month)

        return pd.DataFrame(df)

    @classmethod
    def _operate_on_inputs(cls, func, *inputs) -> ArrayLike:
        return np.apply_along_axis(func, 0, np.stack(inputs))

    @cached_property
    def time_past_reversion(self) -> ArrayLike:
        return self.month_post_reversion + self.forecast_months

    @lru_cache()
    def _time_past_reversion(self, forecast_month: int) -> int:
        return self.time_past_reversion[forecast_month-1]

    @classmethod
    def lookup_value(self, index, df, col: str):
        # Reindexing the DataFrame to handle missing indices
        reindexed_df = df.reindex(index, fill_value=np.nan)
        return reindexed_df.values

    @cached_property
    def cpr(self) -> ArrayLike:
        return np.apply_along_axis(lambda x: self.lookup_value(x, self.input_cpr, 'cpr'), 0, self.time_past_reversion)

    @lru_cache()
    def _cpr(self, forecast_month: int) -> float:
        return self.cpr[forecast_month-1]

    @cached_property
    def cdr(self) -> ArrayLike:
        return np.apply_along_axis(lambda x: self.lookup_value(x, self.input_cdr, 'cdr'), 0, self.time_past_reversion)

    @lru_cache()
    def _cdr(self, forecast_month: int) -> float:
        return self.cdr[forecast_month-1]

    @lru_cache()
    def _remaining_term(self, forecast_month: int) -> int:
        return self._jitted_remaining_term(forecast_month, self.month_to_maturity)

    @staticmethod
    @njit(cache=True)
    def _jitted_remaining_term( forecast_month: int, month_to_maturity: int) -> int:
        return 0.0 if forecast_month > month_to_maturity else (month_to_maturity-forecast_month+1)

    @lru_cache()
    def _opening_balance(self, forecast_month: int) -> float:
        return self.current_balance if forecast_month == 0 else self._jitted_opening_balance(self._closing_balance(forecast_month-1))

    @staticmethod
    @njit(cache=True)
    def _jitted_opening_balance(closing_balance: float) -> float:
        return closing_balance

    @lru_cache()
    def _interest_rate(self, forecast_month: int) -> float:
        return self._jitted_interest_rate(self.boe_base_rate[forecast_month-1], self.post_reversion_margin,
                                          self._time_past_reversion(forecast_month), self.fixed_pre_reversion_rate)

    @staticmethod
    @njit(cache=True)
    def _jitted_interest_rate(boe_base_rate: float, post_reversion_margin: float, time_past_reversion: int, fixed_pre_reversion_rate: float) -> float:
        return boe_base_rate + post_reversion_margin if time_past_reversion > 0 else fixed_pre_reversion_rate

    #cannot be jitted due to pmt
    @lru_cache()
    def _scheduled_payment(self, forecast_month: int) -> float:
        import numpy_financial as npf
        remaining_term = self._remaining_term(forecast_month)
        if remaining_term == 0:
            return 0.0
        balance = self._opening_balance(forecast_month)
        interest_rate = self._interest_rate(forecast_month)
        return npf.pmt(interest_rate/12.0, remaining_term, -balance, balance if self.repayment_method == _INTEREST_ONLY_LITERAL else 0.0)

    @lru_cache()
    def _scheduled_interest(self, forecast_month: int) -> float:
        return self._jitted_scheduled_interest(self._interest_rate(forecast_month), self._opening_balance(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_scheduled_interest(interest_rate: float, opening_balance: float) -> float:
        return interest_rate * (1.0 / 12) * opening_balance

    @lru_cache()
    def _scheduled_principal(self, forecast_month: int) -> float:
        return self._jitted_scheduled_principal(self._scheduled_payment(forecast_month), self._scheduled_interest(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_scheduled_principal(scheduled_payment: float, scheduled_interest: float) -> float:
        return scheduled_payment - scheduled_interest

    @lru_cache()
    def _principal_balloon(self, forecast_month: int) -> float:
        return self._jitted_principal_balloon(self._opening_balance(forecast_month), self._remaining_term(forecast_month), self.repayment_method)

    @staticmethod
    @njit(cache=True)
    def _jitted_principal_balloon(opening_balance: float, remaining_term: int, repayment_method: str) -> float:
        return opening_balance if (remaining_term == 1 and repayment_method == _INTEREST_ONLY_LITERAL) else 0.0

    @lru_cache()
    def _closing_balance(self, forecast_month: int):
        return self._jitted_closing_balance(self._opening_balance(forecast_month), self._scheduled_principal(forecast_month), self._principal_balloon(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_closing_balance(opening_balance: float, scheduled_principal: float, principal_balloon: float) -> float:
        return opening_balance-scheduled_principal-principal_balloon

    @lru_cache()
    def _expected_opening_performing_balance(self, forecast_month: int) -> float:
        return self._opening_balance(forecast_month) if forecast_month == 1 else self._jitted_expected_opening_performing_balance(self._expected_closing_performing_balance(forecast_month-1))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_opening_performing_balance(expected_closing_performing_balance: float) -> float:
        return expected_closing_performing_balance

    @lru_cache()
    def _defaults(self, forecast_month: int) -> float:
        return self._jitted_defaults(self._cdr(forecast_month), self._expected_opening_performing_balance(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_defaults(cdr: float, expected_opening_performing_balance: float) -> float:
        return (1 - (1 - cdr) ** (1.0 / 12)) * expected_opening_performing_balance

    @lru_cache()
    def _expected_balance_post_period_defaults(self, forecast_month: int) -> float:
        return self._jitted_expected_balance_post_period_defaults(self._expected_opening_performing_balance(forecast_month), self._defaults(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_balance_post_period_defaults(expected_opening_performing_balance: float, defaults: float) -> float:
        return expected_opening_performing_balance - defaults

    @lru_cache()
    def _survival_percentage_post_default(self, forecast_month: int) -> float:
        return self._jitted_survival_percentage_post_default(self._opening_balance(forecast_month), self._expected_balance_post_period_defaults(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_survival_percentage_post_default(balance: float, expected_balance_post_period_defaults: float) -> float:
        return (expected_balance_post_period_defaults / balance) if balance else 0.0

    @lru_cache()
    def _expected_scheduled_payment(self, forecast_month: int) -> float:
        return self._jitted_expected_scheduled_payment(self._survival_percentage_post_default(forecast_month), self._scheduled_payment(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_scheduled_payment(survival_percentage_post_default: float, scheduled_payment: float) -> float:
        return survival_percentage_post_default * scheduled_payment

    @lru_cache()
    def _expected_interest(self, forecast_month: int) -> float:
        return self._jitted_expected_interest(self._survival_percentage_post_default(forecast_month), self._scheduled_interest(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_interest(survival_percentage_post_default: float, scheduled_interest: float) -> float:
        return survival_percentage_post_default * scheduled_interest

    @lru_cache()
    def _expected_principal_schedule(self, forecast_month: int) -> float:
        return self._jitted_expected_principal_schedule(self._scheduled_principal(forecast_month), self._principal_balloon(forecast_month), self._survival_percentage_post_default(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_principal_schedule(scheduled_principal: float, principal_balloon: float, survival_percentage_post_default: float) -> float:
        return (scheduled_principal + principal_balloon) * survival_percentage_post_default

    @lru_cache()
    def _expected_balance_pre_period_prepays(self, forecast_month: int) -> float:
        return self._jitted_expected_balance_pre_period_prepays(self._expected_balance_post_period_defaults(forecast_month), self._expected_principal_schedule(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_balance_pre_period_prepays(expected_balance_post_period_defaults: float, expected_principal_schedule: float) -> float:
        return expected_balance_post_period_defaults - expected_principal_schedule

    @lru_cache()
    def _expected_prepayments(self, forecast_month: int) -> float:
        return self._jitted_expected_prepayments(self._cpr(forecast_month), self._expected_balance_pre_period_prepays(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_prepayments(cpr: float, expected_balance_pre_period_prepays: float) -> float:
        return (1 - (1 - cpr) ** (1.0 / 12)) * expected_balance_pre_period_prepays

    @lru_cache()
    def _expected_closing_performing_balance(self, forecast_month: int) -> float:
        return self._jitted_expected_closing_performing_balance(self._expected_balance_pre_period_prepays(forecast_month), self._expected_prepayments(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_closing_performing_balance(expected_balance_pre_period_prepays: float, expected_prepayments: float) -> float:
        return expected_balance_pre_period_prepays - expected_prepayments

    @lru_cache()
    def _end_of_period_survival(self, forecast_month: int) -> float:
        return self._jitted_end_of_period_survival(self._closing_balance(forecast_month), self._expected_closing_performing_balance(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_end_of_period_survival(closing_balance: float, expected_closing_performing_balance: float) -> float:
        return expected_closing_performing_balance / closing_balance if closing_balance else 0.0

    @lru_cache()
    def _expected_opening_default_balance(self, forecast_month: int) -> float:
        return 0.0 if forecast_month == 1 else self._jitted_expected_opening_default_balance(self._expected_closing_default_balance(forecast_month-1))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_opening_default_balance(expected_closing_default_balance: float) -> float:
        return expected_closing_default_balance

    @lru_cache()
    def _expected_new_defaults(self, forecast_month: int) -> float:
        return self._jitted_expected_new_defaults(self._defaults(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_new_defaults(defaults: float) -> float:
        return defaults

    @lru_cache()
    def _expected_recoveries(self, forecast_month: int) -> float:
        return 0.0 if forecast_month <= self.recovery_lag else self._jitted_expected_recoveries(self.ls, self._expected_new_defaults(forecast_month-self.recovery_lag))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_recoveries(ls: float, expected_new_defaults: float) -> float:
        return  (1-ls) * expected_new_defaults

    @lru_cache()
    def _expected_loss(self, forecast_month: int) -> float:
        return 0.0 if forecast_month <= self.recovery_lag else self._jitted_expected_loss(self.ls, self._expected_new_defaults(forecast_month-self.recovery_lag))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_loss(ls: float, expected_new_defaults: float) -> float:
        return ls * expected_new_defaults

    @lru_cache()
    def _expected_closing_default_balance(self, forecast_month: int) -> float:
        return self._jitted_expected_closing_default_balance(self._expected_opening_default_balance(forecast_month),
                                                             self._expected_new_defaults(forecast_month),
                                                             self._expected_recoveries(forecast_month),
                                                             self._expected_loss(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_expected_closing_default_balance(expected_opening_default_balance: float, expected_new_defaults: float, expected_recoveries: float, expected_loss: float) -> float:
        return expected_opening_default_balance + expected_new_defaults - expected_recoveries - expected_loss


BASE_LOAN_CONFIG = {
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


def run_single_simulation(loan_config_override: dict = {}):
    configuration = {**BASE_LOAN_CONFIG, **loan_config_override}
    try:
        model = MortgageCashflowModel(**configuration)
        model.run()
        return model.df
    except Exception as e:
        return f'Error encountered when running simulation with config {loan_config_override}: {e}'


def run_simulations(loan_configs: List[Dict]=[{}]):
    '''
    Alternatively can use Dask, which works well with the data frames too
    '''
    import os
    from multiprocessing import Pool
    with Pool(os.cpu_count()) as p:
        results = (p.map(run_single_simulation, loan_configs))
    return results


if __name__ == '__main__':
    import time
    starting_time = time.time()
    run_simulations([{}] * int(1e4))
    print(time.time() - starting_time)

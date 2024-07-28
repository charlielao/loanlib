import numpy as np
import pandas as pd
from functools import cached_property
from loanlib.utils import shift_by_months, get_first_truth_value

_INTEREST_ONLY_LITERAL = 'Interest Only'
_REPAYMENT_METHODS = {_INTEREST_ONLY_LITERAL}

#can do Similar sort of computational graph so everything only has to be evaluated once automatically, or alternatively caching
#could potentially refactored out all the numpy computations in fact

class MortgageCashflowModel:

    def __init__(self, month_post_reversion: int, seasoning: int, current_balance : float, fixed_pre_reversion_rate: float,
                    post_reversion_margin: float, month_to_maturity: int, repayment_method: str, ls: float, recovery_lag: int,
                    cpr: pd.Series, cdr: pd.Series, boe_base_rate: np.array):
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
        self.input_cpr = cpr
        self.input_cdr = cdr
        self.number_month_forecasted = len(self.boe_base_rate)
        self.forecast_months = np.arange(1, self.number_month_forecasted+1)
        self.verify_parameters()

    def verify_parameters(self):
        if self.fixed_pre_reversion_rate < 0.0 or self.fixed_pre_reversion_rate > 1.0:
            raise ValueError('Fixed pre-reversion rate should be between 0 and 1')

        if self.post_reversion_margin < 0.0 or self.post_reversion_margin > 1.0:
            raise ValueError('Post-reversion margin should be between 0 and 1')

        if self.repayment_method not in _REPAYMENT_METHODS:
            raise NotImplementedError('Only interest only repayment method is supported')

        if self.cpr.index != self.cdr.index:
            raise ValueError('CPR and CDR indices should match')

    @classmethod
    def operate_on_inputs(cls, func, *inputs):
        return np.apply_along_axis(func, 0, np.stack(inputs))

    @cached_property
    def remaining_term(self):
        return np.concatenate((np.arange(1, self.month_to_maturity+1)[::-1],
                               np.zeros(self.number_month_forecasted-self.month_to_maturity)), axis=0)

    @cached_property
    def time_past_reversion(self):
        return self.month_post_reversion + self.forecast_months

    @cached_property
    def opening_balance(self):
        return shift_by_months(self.closing_balance)

    @cached_property
    def interest_rate(self):
        def func(x): return x[1] + self.post_reversion_margin if x[0]>0 else self.fixed_pre_reversion_rate
        return self.operate_on_inputs(func, self.time_past_reversion, self.boe_base_rate)


    @cached_property
    def scheduled_payment(self):
        import numpy_financial as npf
        def func(x): return npf.pmt(x[2]/12.0, x[0], -x[1], x[1] if self.repayment_method == _INTEREST_ONLY_LITERAL else 0.0)
        return self.operate_on_inputs(func, self.remaining_term, self.opening_balance, self.interest_rate)


    @cached_property
    def scheduled_interest(self):
        return self.operate_on_inputs(lambda x: x[1] * (1.0/12) * x[1], self.opening_balance, self.interest_rate)

    @cached_property
    def scheduled_principal(self):
        return self.operate_on_inputs(lambda x: x[0] - x[1], self.scheduled_payment, self.scheduled_interest)

    @cached_property
    def principal_balloon(self):
        def func(x): return x[1] if x[0] == 1 and self.repayment_method == _INTEREST_ONLY_LITERAL else 0.0
        return self.operate_on_inputs(func, self.remaining_term, self.opening_balance)

    @cached_property
    def closing_balance(self):
        return self.operate_on_inputs(lambda x: x[0]-x[1]-x[2], self.opening_balance, self.scheduled_principal, self.principal_balloon)

    @cached_property
    def expected_opening_performing_balance(self):
        return self.operate_on_inputs(lambda x: x[1] if x[0] == 1 else x[2],self.forecast_months, self.opening_balance,
                                                        shift_by_months(self.expected_closing_performing_balance))

    @cached_property
    def cdr(self):
        return np.apply_along_axis(lambda x: self.input_cdr.loc[x], 0, self.time_past_reversion)


    @cached_property
    def defaults(self):
        return self.operate_on_inputs(lambda x: (1-(1-x[1])**(1.0/12))*x[0], self.expected_opening_performing_balance, self.cdr)

    @cached_property
    def expected_balance_post_period_defaults(self):
        return self.operate_on_inputs(lambda x: x[0]-x[1], self.expected_opening_performing_balance, self.defaults)

    @cached_property
    def survival_percentage_post_default(self):
        return self.operate_on_inputs(lambda x: x[1]/x[0] if x[0] else 0.0,self.opening_balance, self.expected_balance_post_period_defaults)

    @cached_property
    def expected_scheduled_payment(self):
        return self.operate_on_inputs(lambda x: x[1]*x[0], self.survival_percentage_post_default, self.scheduled_payment)

    @cached_property
    def expected_interest(self):
        return self.operate_on_inputs(lambda x: x[1]*x[0], self.survival_percentage_post_default, self.scheduled_interest)

    @cached_property
    def expected_principal_schedule(self):
        return self.operate_on_inputs(lambda x: (x[1]+x[2])*x[0],
                                      self.survival_percentage_post_default, self.scheduled_principal, self.principal_balloon)


    @cached_property
    def expected_balance_pre_period_prepays(self):
        return self.operate_on_inputs(lambda x: x[0]-x[1], self.expected_balance_post_period_defaults, self.expected_principal_schedule)

    @cached_property
    def cpr(self):
        return np.apply_along_axis(lambda x: self.input_cpr.loc[x], 0, self.time_past_reversion)

    @cached_property
    def expected_prepayments(self):
        return self.operate_on_inputs(lambda x: (1-(1-x[0])**(1.0/12))*x[1], self.cpr, self.expected_balance_pre_period_prepays)

    @cached_property
    def expected_closing_performing_balance(self):
        return self.operate_on_inputs(lambda x: x[0]-x[1], self.expected_balance_pre_period_prepays, self.expected_prepayments)

    @cached_property
    def end_of_period_survival(self):
        return self.operate_on_inputs(lambda x: x[0]/x[1] if x[1] else 0.0,self.expected_closing_performing_balance, self.closing_balance)

    @cached_property
    def expected_opening_default_balance(self):
        return self.operate_on_inputs(lambda x: 0.0 if x[0] == 1 else x[1], self.forecast_months, shift_by_months(self.expected_closing_default_balance))

    @cached_property
    def expected_new_defaults(self):
        return self.defaults

    @cached_property
    def expected_recoveries(self):
        def func(x): return 0.0 if x[1] <= self.recovery_lag else (1-self.ls) * x[0]
        return self.operate_on_inputs(func, shift_by_months( self.expected_new_defaults, self.recovery_lag), self.forecast_months)

    @cached_property
    def expected_loss(self):
        def func(x): return 0.0 if x[1] <= self.recovery_lag else self.ls * x[0]
        return self.operate_on_inputs(func, shift_by_months( self.expected_new_defaults, self.recovery_lag), self.forecast_months)

    @cached_property
    def expected_closing_default_balance(self):
        return self.operate_on_inputs(lambda x: x[0]-x[2]-x[3]+x[1],
            self.expected_opening_default_balance, self.expected_new_defaults, self.expected_recoveries, self.expected_loss)

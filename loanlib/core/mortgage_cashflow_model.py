_REPAYMENT_METHODS = {'Interest Only'}


class MortgageCashflowModel:

    def __init__(self, month_post_reversion: int, seasoning: int, current_balance : float, fixed_pre_reversion_rate: float,
                    post_reversion_margin: float, month_to_maturity: int, repayment_method : str):
        self.month_post_reversion = month_post_reversion
        self.seasoning = seasoning
        self.current_balance = current_balance
        self.fixed_pre_reversion_rate = fixed_pre_reversion_rate
        self.post_reversion_margin = post_reversion_margin
        self.month_to_maturity = month_to_maturity
        self.repayment_method = repayment_method
        self.verify_parameters()


    def verify_parameters(self):
        if self.fixed_pre_reversion_rate < 0.0 or self.fixed_pre_reversion_rate > 1.0:
            raise ValueError('Fixed pre-reversion rate should be between 0 and 1')

        if self.post_reversion_margin < 0.0 or self.post_reversion_margin > 1.0:
            raise ValueError('Post-reversion margin should be between 0 and 1')

        if self.repayment_method not in _REPAYMENT_METHODS:
            raise NotImplementedError('Only interest only repayment method is supported')


    def remaining_term(self):
        pass

    def time_past_reversion(self):
        pass

    def opening_balance(self):
        pass

    def interest_rate(self):
        pass

    def scheduled_payment(self):
        pass

    def scheduled_interest(self):
        pass

    def scheduled_principal(self):
        pass

    def principal_balloon(self):
        pass

    def closing_balance(self):
        pass

    def expected_opening_performing_balance(self):
        pass

    def cdr(self):
        # Implementation here
        pass

    def defaults(self):
        # Implementation here
        pass

    def expected_balance_post_period_defaults(self):
        # Implementation here
        pass

    def survival_percentage_post_default(self):
        # Implementation here
        pass

    def expected_scheduled_payment(self):
        # Implementation here
        pass

    def expected_interest(self):
        # Implementation here
        pass

    def expected_principal_schedule(self):
        # Implementation here
        pass

    def expected_balance_pre_period_prepays(self):
        # Implementation here
        pass

    def cpr(self):
        # Implementation here
        pass

    def expected_prepayments(self):
        # Implementation here
        pass

    def expected_closing_performing_balance(self):
        # Implementation here
        pass

    def end_of_period_survival(self):
        # Implementation here
        pass

    def expected_opening_default_balance(self):
        # Implementation here
        pass

    def expected_new_defaults(self):
        # Implementation here
        pass

    def expected_recoveries(self):
        # Implementation here
        pass

    def expected_loss(self):
        # Implementation here
        pass

    def expected_closing_default_balance(self):
        # Implementation here
        pass

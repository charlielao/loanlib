import pytest
from test_data import _get_data
from loanlib.core.model import run_single_simulation, BASE_LOAN_CONFIG


class TestModel:

    @pytest.fixture(autouse=True)
    def _get_model(self):
        self.df = run_single_simulation()

    def test_expected_opening_performing_balance(self):
        assert all((val >= 0.0 and val <= BASE_LOAN_CONFIG[ 'current_balance' ] for val in self.df['expected_opening_performing_balance'].values))

    def test_expected_balance_post_period_defaults(self):
        assert all((val >= 0.0 and val <= BASE_LOAN_CONFIG[ 'current_balance' ] for val in self.df['expected_balance_post_period_defaults'].values))

    def test_survival_percentage_post_default(self):
        assert all((val >= 0.0 and val <= 1.0 for val in self.df['survival_percentage_post_default'].values))

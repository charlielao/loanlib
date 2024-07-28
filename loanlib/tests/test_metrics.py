import pytest
from test_data import _get_data
from loanlib.core.loan_metrics import LoanMetrics


class TestMetrics:

    @pytest.fixture(autouse=True)
    def _get_curves(self, _get_data):
        self.curves = LoanMetrics(_get_data)

    def test_curves(self):
        for metric in ['SMM', 'MDR', 'CPR', 'CDR']:
            assert all((val <= 1.0 for val in self.curves.curve(metric).values))

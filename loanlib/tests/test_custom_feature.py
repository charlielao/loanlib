from loanlib.core import custom_feature
import pytest
import pandas as pd
from datetime import date


@pytest.fixture
def example_portfolio():
    dates = (date(2020, 8, 31), date(2020, 9, 30), date(2020, 10, 31), date(2020, 11, 30), date(2020, 12, 31),
             date(2021, 1, 31), date(2021, 2, 28), date(2021,3, 31))
    origination_date = ()
    return pd.DataFrame()


def test_year_of_default():
    custom_feature.year_of_default()

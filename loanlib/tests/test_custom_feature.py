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


def small_portfolio():
    SOURCE_FILE_PATH = '../../data/2024 - Strat Casestudy.xlsx'
    from loanlib.data_loader import DataLoader
    import random
    random.seed(10)
    loader = DataLoader(SOURCE_FILE_PATH)
    df = loader.combined_data_frame
    all_loans = set(index[0] for index in df.index)
    random_set = random.sample(list(all_loans),10)
    return loader.create_features(df[df.index.get_level_values('ID').isin(random_set)])


class TestCustomeFeatures:

    @pytest.fixture(autouse=True)
    def _get_data(self):
        self._df = small_portfolio()

    def test_year_of_default(self):
        assert len(self._df[ 'year_of_default' ]) > 0

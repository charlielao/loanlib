from test_data import _get_data


class TestCustomeFeatures:

    def test_postdefault_recoveries(self, _get_data):
        assert all((val >= 0.0 for val in _get_data[ 'postdefault_recoveries' ].values))

    def test_n_missed_payments(self, _get_data):
        assert all((val >= 0 for val in _get_data[ 'n_missed_payments' ].values))

    def test_default_in_month(self, _get_data):
        #there cannot be more than one truth value
        assert sum((val for val in _get_data[ 'default_in_month' ].values)) <= 1
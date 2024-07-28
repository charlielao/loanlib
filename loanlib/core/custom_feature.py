import datetime

import pandas as pd
from numba import njit
from functools import wraps
import numpy as np

custom_column_register = {}
'''
There are three modes of creating your own features:
1. If the code is easily vectorisable, then you can define a pure numpy function that computes the feature;
2. if it is more complex iterative code, then you can simply add a @njit decorator and it will be compiled to C codes
3. lastly if it is less numerical and require objects such as datetime, then the above don't work really well so you can 
pass in the entire dataframe and compute as normal

The first two ways are more recommended as in theory they should be faster but needs more testing

The first two ways require you specify the column names of the input features in the same order of the arguments so the 
decorator will transform the dataframe input into numpy arrays as numba is not fast on pure pandas objects 

The arguments in the decorator also defines the dependencies so that all the feature functions can be defined in any order
and the computational graph will automatically be traced and computed in the correct order

for future iteration, there should be something that intercept the call and construct the computational graph automatically
without user specifiying the dependencies explicityly

'''


def custom_feature(*column_names, **kwargs):
    def decorator(func):
        import inspect
        custom_column_register[func.__name__] = column_names
        signatures = inspect.get_annotations(func)
        if any( (func_type == pd.core.frame.DataFrame for func_type in signatures.values()) ):
            return func

        @wraps(func)
        def wrapper(df: pd.DataFrame):
            columns_as_lists = [np.array(df[col]) for col in column_names]
            return pd.Series(func(*columns_as_lists))
        return wrapper
    return decorator


def fill_static(fill_array, reference_array, is_date: bool=True):
    idx = np.argmax(reference_array)
    return np.full(shape=len(reference_array), fill_value=(np.datetime64(fill_array[idx]) if is_date else fill_array[idx])
                                                        if reference_array[idx] else None)


def get_static_value(values):
    return values[0]


@custom_feature('Payment Made', 'original_balance')
def current_balance(payments_made: np.array, balances: np.array):
    return np.clip(balances - np.cumsum(payments_made), a_min=0.0, a_max=None)


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


@custom_feature()
def seasoning(df: pd.DataFrame):
    return df.apply(lambda row : diff_month(row['Date'], row['origination_date']), axis=1)


@custom_feature('Payment Made', 'Payment Due')
@njit
def missed_payments(payment_made: np.array, payment_due: np.array):
    results = np.zeros(len(payment_made), dtype=np.bool_)
    for index, (made, due) in enumerate(zip(payment_made, payment_due)):
        results[index] = (made < due)
    return results


@custom_feature('missed_payments')
@njit
def n_missed_payments(missed_payments: np.array):
    curr_count = 0
    results = np.zeros(len(missed_payments))
    for index, missed in enumerate(missed_payments):
        if missed:
            curr_count += 1
        else:
            curr_count = 0
        results[index] = curr_count
    return results


@custom_feature('Payment Made', 'Payment Due', 'current_balance')
@njit
def prepaid_in_month(payment_made: np.array, payment_due: np.array, current_balance:np.array):
    results = np.zeros(len(current_balance), dtype=np.bool_)
    for index, (p_made, p_due, balance) in enumerate(zip(payment_made, payment_due, current_balance)):
        results[index] = p_made > p_due and balance < 1e-8
    return results


@custom_feature('n_missed_payments')
@njit
def default_in_month(missed_payments: np.array):
    results = np.zeros(len(missed_payments), dtype=np.bool_)
    has_default = False
    for index, missed in enumerate(missed_payments):
        has_default = has_default or missed >= 3
        results[index] = has_default
    return results


@custom_feature('is_recovery_payment')
@njit
def recovery_in_month(recovery_payment: np.array):
    results = np.zeros(len(recovery_payment), dtype=np.bool_)
    has_recovery = False
    for index, recovery in enumerate(recovery_payment):
        has_recovery = has_recovery or recovery
        results[index] = has_recovery
    return results


@custom_feature('Date', 'date_of_default', 'Payment Made')
def is_recovery_payment(dates: np.array, date_of_default: np.array, payment_made: np.array):
    if np.isnan(date_of_default).any():
        return np.full(shape=len(date_of_default), fill_value=False)
    else:
        default_date = get_static_value(date_of_default)
        applicable_dates = np.apply_along_axis(lambda x: x>default_date, 0, dates.astype('datetime64'))
        return np.logical_and(applicable_dates, np.apply_along_axis(lambda x:x>0.0, 0, payment_made))


@custom_feature()
def time_to_reversion(df: pd.DataFrame):
    return df.apply(lambda row : diff_month(row['reversion_date'], row['Date']), axis=1)


@custom_feature('investor_1_acquisition_date', 'Date')
def is_post_seller_purchase_date(acquisition_date: np.array, dates: np.array):
    acquisition_date = acquisition_date[0]
    return np.apply_along_axis(lambda x: x >= acquisition_date, 0, dates.astype('datetime64'))

@custom_feature('Payment Made', 'is_recovery_payment')
@njit
def postdefault_recoveries(payment_made: np.array, is_recovery_payment: np.array):
    cum_recovery = 0.0
    for index, (payment, is_recovery) in enumerate(zip(payment_made, is_recovery_payment)):
        cum_recovery += payment if is_recovery else 0.0
    return np.full(shape=len(payment_made), fill_value=cum_recovery)


@custom_feature('Date', 'prepaid_in_month')
def prepayment_date(dates: np.array, prepaids: np.array):
    return fill_static(dates, prepaids)


@custom_feature('Date', 'default_in_month')
def date_of_default(dates: np.array, default_in_month: np.array):
    return fill_static(dates, default_in_month)


@custom_feature('Date', 'is_recovery_payment')
def date_of_recovery(dates: np.array, is_recovery_payment: np.array):
    return fill_static(dates, is_recovery_payment)


@custom_feature('Month End Balances', 'default_in_month')
def exposure_at_default(month_end_balances: np.array, default_in_month: np.array):
    return fill_static(month_end_balances, default_in_month, False)


@custom_feature('postdefault_recoveries', 'exposure_at_default')
def recovery_percent(postdefault_recoveries: np.array, exposure_at_default: np.array):
    return np.full(len(postdefault_recoveries), get_static_value(postdefault_recoveries) / get_static_value(exposure_at_default) )


@custom_feature('date_of_default', 'Date')
def months_since_default(df: pd.DataFrame):
    return df.apply(lambda row : diff_month(row['date_of_default'], row['Date']), axis=1)


@custom_feature('date_of_default')
def year_of_default( df: pd.DataFrame):
    return df['date_of_default'].apply(lambda x: x.year)
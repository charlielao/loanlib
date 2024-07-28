import datetime

import pandas as pd
from numba import njit
from functools import wraps
import numpy as np
from numpy.typing import ArrayLike
from loanlib.utils import get_first_truth_value, fill_static

custom_column_register = {}
'''
There are three modes of creating your own features:
1. If the code is easily vectorisable, then you can define a pure numpy function that computes the feature;
2. if it is more complex iterative code, then you can simply add a @njit decorator and it will be compiled to C codes
3. lastly if it is less numerical and require objects such as datetime, then the above don't work really well so you can 
pass in the entire dataframe and compute as normal

The first two ways are more recommended as in theory they should be faster but needs more testing to confirm

The first two ways require you specify the column names of the input features in the same order of the arguments so the 
decorator will transform the dataframe input into numpy arrays as numba is not fast on pure pandas objects 

The arguments in the decorator also defines the dependencies so that all the feature functions can be defined in any order
and the computational graph will automatically be traced and computed in the correct order

for future iteration, there should be something that intercept the call and construct the computational graph automatically
without user specifying the dependencies explicitly; also possbily don't need to specify both in arguments and in the decorator
but inferred from the function signature 

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


def get_static_value(values):
    return values[0]


@custom_feature('Payment Made', 'original_balance')
def current_balance(payments_made: ArrayLike, balances: ArrayLike) -> ArrayLike:
    '''
    The current balance outstanding for each loan and month.
    '''
    return np.clip(balances - np.cumsum(payments_made), a_min=0.0, a_max=None)


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


@custom_feature()
def seasoning(df: pd.DataFrame) -> pd.DataFrame:
    '''
    the integer number of months since the loan was originated at each month.
    '''
    return df.apply(lambda row : diff_month(row['Date'], row['origination_date']), axis=1)


@custom_feature('Payment Made', 'Payment Due')
@njit
def missed_payments(payment_made: ArrayLike, payment_due: ArrayLike) -> ArrayLike:
    '''
    whether this payment is missed
    '''
    results = np.zeros(len(payment_made), dtype=np.bool_)
    for index, (made, due) in enumerate(zip(payment_made, payment_due)):
        results[index] = (made < due)
    return results


@custom_feature('missed_payments')
@njit
def n_missed_payments(missing_payments: ArrayLike) -> ArrayLike:
    '''
    number of missed payments in a row.
    '''
    curr_count = 0
    results = np.zeros(len(missing_payments))
    for index, missed in enumerate(missing_payments):
        if missed:
            curr_count += 1
        else:
            curr_count = 0
        results[index] = curr_count
    return results


@custom_feature('Payment Made', 'Payment Due', 'current_balance')
@njit
def prepaid_in_month(payment_made: ArrayLike, payment_due: ArrayLike, balance: ArrayLike) -> ArrayLike:
    '''
    a flag indicating that the borrower prepaid in a given month.
    '''
    results = np.zeros(len(balance), dtype=np.bool_)
    for index, (p_made, p_due, balance) in enumerate(zip(payment_made, payment_due, balance)):
        results[index] = p_made > p_due and balance < 1e-8
    return results


@custom_feature('n_missed_payments')
@njit
def default_in_month(missing_payments: ArrayLike) -> ArrayLike:
    '''
    a flag indicating that the borrower defaulted in a given month.
    '''
    results = np.zeros(len(missing_payments), dtype=np.bool_)
    has_default = False
    for index, missed in enumerate(missing_payments):
        has_default = has_default or missed >= 3
        results[index] = has_default
    return results


@custom_feature('is_recovery_payment')
@njit
def recovery_in_month(recovery_payment: ArrayLike) -> ArrayLike:
    '''
    a flag indicating that a recovery has been made post-default in a given month.
    '''
    results = np.zeros(len(recovery_payment), dtype=np.bool_)
    has_recovery = False
    for index, recovery in enumerate(recovery_payment):
        has_recovery = has_recovery or recovery
        results[index] = has_recovery
    return results


@custom_feature('Date', 'date_of_default', 'Payment Made')
def is_recovery_payment(dates: ArrayLike, default_date: ArrayLike, payment_made: ArrayLike) -> ArrayLike:
    '''
    a flag indicating whether the associated payment has been made post-default.
    '''
    if np.isnan(default_date).any():
        return np.full(shape=len(default_date), fill_value=False)
    else:
        default_date = get_static_value(default_date)
        applicable_dates = np.apply_along_axis(lambda x: x>default_date, 0, dates.astype('datetime64'))
        return np.logical_and(applicable_dates, np.apply_along_axis(lambda x:x>0.0, 0, payment_made))


@custom_feature()
def time_to_reversion(df: pd.DataFrame) -> pd.DataFrame:
    '''
    The integer number of months until the laon reverts. This is negative if the loan is before reversion and 0 at the month of reversion.
    '''
    return df.apply(lambda row : diff_month(row['reversion_date'], row['Date']), axis=1)


@custom_feature('investor_1_acquisition_date', 'Date')
def is_post_seller_purchase_date(acquisition_date: ArrayLike, dates: ArrayLike) -> ArrayLike:
    '''
    Is this time period after the seller purchased this loan.
    '''
    acquisition_date = acquisition_date[0]
    return np.apply_along_axis(lambda x: x >= acquisition_date, 0, dates.astype('datetime64'))


@custom_feature('Payment Made', 'is_recovery_payment')
@njit
def postdefault_recoveries(payment_made: ArrayLike, recovery_payment_flag: ArrayLike) -> ArrayLike:
    '''
    The cumulative recoveries post-default.
    '''
    cum_recovery = 0.0
    for index, (payment, is_recovery) in enumerate(zip(payment_made, recovery_payment_flag)):
        cum_recovery += payment if is_recovery else 0.0
    return np.full(shape=len(payment_made), fill_value=cum_recovery)


@custom_feature('Date', 'prepaid_in_month')
def prepayment_date(dates: ArrayLike, prepaids: ArrayLike) -> ArrayLike:
    '''
    the date that the loan prepays (or nan if it does not).
    '''
    idx = get_first_truth_value(prepaids)
    return fill_static(dates[idx] if idx>0 else None, len(prepaids))


@custom_feature('Date', 'default_in_month')
def date_of_default(dates: ArrayLike, default_month_flag: ArrayLike) -> ArrayLike:
    '''
    the date that the loan defaults (or nan if it does not).
    '''
    idx = get_first_truth_value(default_month_flag)
    return fill_static(dates[idx] if idx > 0 else None, len(default_month_flag))


@custom_feature('Date', 'is_recovery_payment')
def date_of_recovery(dates: ArrayLike, recovery_payment_flag: ArrayLike) -> ArrayLike:
    '''
    the date that a recovery is made on the loan, post-default.
    '''
    idx = get_first_truth_value(recovery_payment_flag)
    return fill_static(dates[idx] if idx > 0 else None, len(recovery_payment_flag))


@custom_feature('current_balance', 'default_in_month')
def exposure_at_default(balances: ArrayLike, default_month: ArrayLike) -> ArrayLike:
    '''
    the current balance of the loan outstanding at default.
    '''
    idx = get_first_truth_value(default_month)
    return fill_static(balances[idx] if idx > 0 else None, len(default_month), False)


@custom_feature('postdefault_recoveries', 'exposure_at_default')
def recovery_percent(recoveries: ArrayLike, exposures: ArrayLike) -> ArrayLike:
    '''
    the postdefault_recoveries as a percentage of the exposure at default.
    '''
    return fill_static(get_static_value(recoveries) / get_static_value(exposures), len(recoveries), False)


@custom_feature('date_of_default', 'Date')
def months_since_default(df: pd.DataFrame) -> pd.DataFrame:
    '''
    the number months since default happend
    '''
    return df.apply(lambda row : diff_month(row['date_of_default'], row['Date']), axis=1)


@custom_feature('date_of_default')
def year_of_default(df: pd.DataFrame) -> pd.DataFrame:
    '''
    the year defaults happend
    '''
    return df['date_of_default'].apply(lambda x: x.year)


@custom_feature('date_of_default', 'default_in_month', 'Payment Made', 'Payment Due')
def defaulted_amounts(default_dates: ArrayLike, default_in_months_flag: ArrayLike, payments_made: ArrayLike, payments_due: ArrayLike) -> ArrayLike:
    '''
    how much accumulated default amounts there are at each month
    :param default_dates:
    :param default_in_months_flag:
    :param payments_made:
    :param payments_due:
    :return:
    '''
    defaulted_date = get_static_value(default_dates)
    if np.isnan(defaulted_date).any():
        return fill_static(0.0, len(default_dates), False)
    idx = get_first_truth_value(default_in_months_flag)-2#the third day
    results = np.zeros(len(default_dates))
    results[idx:] = np.cumsum(payments_due[idx:]) - np.cumsum(payments_made[idx:])
    return results


@custom_feature('defaulted_amounts', 'current_balance', 'Payment Made', 'Payment Due')
@njit
def outstanding_balance(default_amounts: ArrayLike, balances: ArrayLike, payments_made: ArrayLike, payments_due: ArrayLike) -> ArrayLike:
    '''
    the total outstanding balance that has not defaulted or prepaid
    '''
    results = np.zeros(len(balances))
    for index, (default_amount, balance, payment_made, payments_due) in enumerate(zip(default_amounts, balances, payments_made, payments_due)):
        #end of month balance adds back the prepaid amount and subtract the default amount
        results[index] = balance + ( payment_made - payments_due ) - default_amount
    return results


@custom_feature('prepaid_in_month', 'Payment Made', 'Payment Due', 'outstanding_balance', 'effective_interest_rate')
def smm(prepaid_month: ArrayLike, payments_made: ArrayLike, payments_due: ArrayLike,
            balances: ArrayLike, effective_interest_rate: ArrayLike) -> ArrayLike:
    '''
    the single monthly mortality rate
    :param prepaid_month:
    :param payments_made:
    :param payments_due:
    :param balances:
    :return:
    '''
    if np.isnan(prepaid_month).any():
        return fill_static(0.0, len(prepaid_month), False)
    return np.divide( (payments_made - payments_due) / (1 + effective_interest_rate ), balances, where=(abs(balances)>1e-3))


@custom_feature('date_of_default', 'Payment Made', 'Payment Due', 'outstanding_balance')
def mdr(default_date: ArrayLike, payments_made: ArrayLike, payments_due: ArrayLike, balances: ArrayLike) -> ArrayLike:
    '''
    the monthly default rate
    :param default_date:
    :param payments_made:
    :param payments_due:
    :param balances:
    :return:
    '''
    if np.isnan(default_date).any():
        return fill_static(0.0, len(default_date), False)
    return np.divide( (payments_due - payments_made) , balances, where=(abs(balances) >1e-3))


@custom_feature('recovery_percent')
def recovery(recover_percent: ArrayLike) -> ArrayLike:
    '''
    the recovery rate
    :param recover_percent:
    :return:
    '''
    return recover_percent


@custom_feature('time_to_reversion', 'pre_reversion_fixed_rate', 'post_reversion_boe_margin')
def effective_interest_rate(time_to_reversion: ArrayLike, pre_reversion_rate: ArrayLike, post_reversion_margin: ArrayLike) -> ArrayLike:
    '''
    the effective interest rate
    :param time_to_reversion:
    :param pre_reversion_rate:
    :param post_reversion_margin:
    :return:
    '''
    return np.where(time_to_reversion < 0, pre_reversion_rate, pre_reversion_rate + post_reversion_margin)

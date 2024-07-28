the library contains several pieces:
1. data_handler.py that loads data and automatically constructs features from custom_feature.py (in a new copy)
```
#load the data like this
from loanlib.data_handler import DataLoader, create_features 
loader = DataLoader(SOURCE_FILE_PATH)
df = loader.combined_data_frame
loader.create_features(df)
```
3. creating your own features to add to the data frame is simple, there are two main ways

  I. (more recommended) add numba @njit for complex recursive operations that don't cannot get vectorised easily, either wise use numpy operations 
```
    @custom_feature('input_1', 'input_2')
    @njit
    def custom_feature_1(arr1: ArrayLike, arr2: ArrayLike) -> ArrayLike:
        results = arr1+arr2
        return results
```
  II. alternatively if it's not possible, you can still pass in dataframe, meant for dealing with objects like datetime like don't work well in numpy/numba
```
    @custom_feature()
    def custom_feature_2(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(some_func, axis=1)
```
  in the first case, the order of inputs in the @custom_feature decorator is important as it transforms from pandas object to arrays for you (numba cannot deal with pandas),
  it also provides dependency information that helps automatically determine the order of execution so that you can define the feature in any order and the library 
  will generate them in the correct order
  
4. load_metrics.py provides LoanMetrics class that constructs loan metrics curves (SMM, MDR, CPR, CDR, Recovery) and provides plotting and pivot tables
```
#can pass in df, or just the data source, it will automatically load data and run the create features routine
from loanlib.core.loan_metrics import LoanMetrics

curves = LoanMetrics(df, index='time_to_reversion', pivots=['product'])
curves.curve('CPR')
curves.curve('SMM')

curves2 = LoanMetrics(SOURCE_FILE_PATH)
curves2.plot('CDR')


```
5. model.py provides a simple implementation of the cashflow model that takes can configuration as a dictionary, to modify the model or add a row
   simply provide a pair of functions in this form, currently as there are nonvectorisable recursive functions, we use numba to iterate through functions quickly
```
    @lru_cache()
    def _new_row(self, forecast_month: int) -> float:
        return self._jitted_new_row(self._other_row_1(forecast_month), self._other_row_2(forecast_month))

    @staticmethod
    @njit(cache=True)
    def _jitted_new_row(other_row_1: float, other_row_2: float) -> float:
        return other_row_1 - other_row_2
```
6. lastly, to run cashflow models with run_simulations, which uses multiprocessing libraries to parallel workflow; on my laptop 10000 basic loans take about 15 seconds to simulate
```
from loanlib.core.model import run_simulations

base_curves = LoanMetrics(SOURCE_FILE_PATH, index ='time_to_reversion')
base_cpr = base_curves.curve('CPR')
base_cdr = base_curves.curve('CDR')
config1 = {'cpr': base_cpr, 'cdr':base_cdr}
base_cpr_2['cpr'] = base_cpr.reset_index().apply(lambda x:x['cpr'] * (2.0 if x['time_to_reversion']>=0 else 1.0 ), axis=1).values
config2 = {'cpr': base_cpr_2, 'cdr':base_cdr}
run_simulations([config1, config2])
```
7. there's also a testing package that's work in progress

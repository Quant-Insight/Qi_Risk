#######################################################################################################################################################################################################################################################
# 
# This function calculates the return attribution over predefined periods of time.
#
# Requirements:
#               import pandas
#               import Qi_risk_wrapper
#               import numpy as np
#               from datetime import datetime, timedelta
#               from dateutil.relativedelta import relativedelta
#
# Inputs: 
#               * date_from [date] (optional) - Start date of data required 'YYYY-MM-DD' format.
#               * date_to [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * risk_model[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#
# 
# Output: 
#               * DataFrame - Factor, Specific and Actual return attribution using the predifined periods. 
#
#                                1w        1m        3m         6m       12m
#                Factor    0.309970 -2.199163 -3.391135  -0.418873   7.97200
#                Specific -0.481109  4.673622  9.970228  18.965672  30.66914
#                Actual   -0.171140  2.474458  6.579093  18.546799  38.64114
#
#
########################################################################################################################################################################################################################################################

from Qi_risk_wrapper import RiskData

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def convert_to_bussiness_date(date: datetime) -> str:
    # Transform date to datetime object

    if date.weekday() == 5:
        date = date - timedelta(days=1)
    elif date.weekday() == 6:
        date = date - timedelta(days=2)
        
    return date.strftime('%Y-%m-%d')

def get_multiperiod_dates(date_to: str) -> dict:

    # Define the periods
    periods = {'1w': 1, '1m': 1, '3m': 3, '6m': 6, '12m': 12}

    # Transform date to datetime object
    end_date = datetime.strptime(date_to, '%Y-%m-%d')

    new_dates = {}

    for period in periods.keys():
       if '1w' in period:
           new_date = end_date - relativedelta(weeks=1)    
       else: 
           new_date = end_date - relativedelta(months=periods[period])
        
       new_dates[period] = convert_to_bussiness_date(new_date)

    return new_dates

if __name__ == "__main__":

    model = 'QI_US_MACRO_MT_1'
    asset = 'SPY'
    date_from = '2024-01-01'
    date_to = '2024-10-22'
    api_key = 'YOUR-API-KEY'

    try:
        attribution_df = pd.DataFrame()
        periods = get_multiperiod_dates(date_to)

        for period in periods.keys():
            risk_data = RiskData(model, asset, periods[period], date_to, api_key)
            attribution_df[period] = risk_data.get_cumulative_attribution()[['factor', 'specific', 'actual']].rename(columns={'factor': 'Factor', 'specific': 'Specific', 'actual': 'Actual'}).T[date_to]

        print(attribution_df)

    except Exception as e:
        print(f'Unexpected error: {e}')

#############################################################################################################################################################################################
#
# run_risk_model(factor_set, daily_returns, timeseries)
# is a QI API endpoint to retrieve all risk models available for a specific identifier.
# 
# Inputs:
#               * factor_set[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#               * timeseries [dict] (required) - time series to generate risk data.                                            	
#                                                {
#                                                "timeseries": {
#                                                "2023-11-06": 79.13529524,
#                                                "2023-11-07": 80.46609841,
#                                                ...
#                                                "2025-10-03": 146.5317237
#                                                }
#
#                                                ** Note that if the time series are expressed in returns, they should be expressed in percentage points. 
#
#               * daily_returns [bool] (optional) - If True, the time series consists of daily returns (not price data). By default, this field will be set to False.  
#                                                  
# Output: dict 
#               * "factors": List of factor names (strings), e.g., ["Real Rates", "Metals", "Risk Aversion", ...].
#
#               * "result": An array of objects, each representing daily risk metrics for the analyzed time series. The array is sorted in reverse chronological 
#                           order (most recent date first). Each object contains:
#
#                       * "date": The date in YYYY-MM-DD format (string).
#                       * "total_return": The total return for the day (number - %).
#                       * "factor_return": The return attributed to macro factors (number - %).
#                       * "specific_return": The idiosyncratic (specific) return not explained by factors (number - %).
#                       * "total_risk": The total risk/volatility (number - %).
#                       * "factor_risk": The risk attributed to macro factors (number - %).
#                       * "specific_risk": The idiosyncratic risk (number - %).
#                       * "exposures": An array of exposure values corresponding to the factors (numbers - %).
#                       * "constant": A constant term in the model (number).
#                
#############################################################################################################################################################################################


import qi_client
from qi_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: Qi API Key
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = 'YOUR_API_KEY'

# Uncomment to use proxy - please refer to Connectivity Guidance doc for more details.
# configuration.proxy = 'http://corporateproxy.business.com:8080'

# Instantiate API class.
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

factor_set = "QI_GLOBAL_MACRO_MT_1"

result = api_instance.run_risk_model(
    factor_set=factor_set, daily_returns = False, timeseries=timeseries
)

# Extract the factors list
factors = result['factors']
result_list = result['result']

# DataFrame 1: Returns and Risk metrics
df_risk_data = pd.DataFrame([
    {
        'date': day['date'],
        'total_return': day['total_return'],
        'factor_return': day['factor_return'],
        'specific_return': day['specific_return'],
        'total_risk': day['total_risk'],
        'specific_risk': day['specific_risk']
    }
    for day in result_list
])

# Convert date to datetime if needed
df_risk_data['date'] = pd.to_datetime(df_risk_data['date'])

# DataFrame 2: Factor Exposures
# Create a list of dictionaries where each dict has date + factor exposures
exposures_data = []

for day in result_list:
    row = {'date': day['date']}
    # Map each exposure value to its corresponding factor name
    for factor, exposure in zip(factors, day['exposures']):
        row[factor] = exposure
    exposures_data.append(row)

factor_exposures_df = pd.DataFrame(exposures_data)
factor_exposures_df['date'] = pd.to_datetime(factor_exposures_df['date'])
factor_exposures_df = factor_exposures_df.set_index('date')

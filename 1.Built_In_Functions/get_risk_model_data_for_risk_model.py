#############################################################################################################################################################################################

# get_risk_model_data_for_risk_model(risk_model, instrument/identifier, date_from=date_from, date_to=date_to)
# is a QI API endpoint to retrieve risk model data for a specific instrument or identifier during a given time frame (max 1 year per query).
# 
# Inputs:
#               * date_from [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * date_to [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * risk_model[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#               * instrument[str] or identifier[str] (required) - The instrument or identifier you want to pull data for. For instrument use Qi's default asset naming convention, e.g. 'AAPL'.
#                                                                 For identifier use asset's SEDOL, ISIN or Bloomberg Ticker.
#
# If your equity focus consists of approximately 90% US or Canadian assets, please use the US Model with or without the market. All securities have Global and US models.
#
# Output: Dictionary
# Units: %
#               * e.g.
#                    
#                    {'2023-01-02': {'factor_return': -0.0459811,
#                                    'factor_risk': 1.11728,
#                                    'specific_return': 0.0459811,
#                                    'specific_risk': 0.693031,
#                                    'total_return': 0.0,
#                                    'total_risk': 1.31476},
#                     '2023-01-03': {'factor_return': -0.472153,
#                                    'factor_risk': 1.10973,
#                                    'specific_return': 0.0511612,
#                                    'specific_risk': 0.687539,
#                                    'total_return': -0.420992,
#                                    'total_risk': 1.30545},#
#                                     ...
#                    }
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


risk_model = 'QI_US_MACRO_MT_1' 
date_from = '2023-01-01'  
date_to = '2023-12-31'
instrument='SPY'


try:
    api_response = api_instance.get_risk_model_data_for_risk_model(risk_model, instrument = instrument, date_from = date_from, date_to = date_to)
    pprint(api_response)
except ApiException as e:
    print(f"Exception when calling DefaultApi:get_descriptor_stds_for_risk_model: {e}")

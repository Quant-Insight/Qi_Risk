#############################################################################################################################################################################################

# get_risk_model_data_for_risk_model(risk_model, instrument, date_from=date_from,date_to=date_to)
# is a QI API endpoint to retrieve risk model data for a specific instrument during a given time frame.
# 
# Inputs:
#               * date_from [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * date_to [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * risk_model[str] (required) - The Factor Set e.g 'QI_US_MACRO_MT_1'
#               * instrument[str] (required) - The instrument that you want to pull data from e.g 'SPY'
# Output: Dictionary
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

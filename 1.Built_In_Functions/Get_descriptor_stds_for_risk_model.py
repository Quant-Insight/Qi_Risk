#############################################################################################################################################################################################
# 
# get_descriptor_stds_for_risk_model(risk_model, date_from=date_from, date_to=date_to)
# is a Qi Risk API endpoint to retrieve the standard deviations for factors within the factor set for a given time frame.
# 
# Inputs:
#               * date_from [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * date_to [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * risk_model[str] (required) - The Factor Set e.g 'QI_US_MACRO_MT_1'
# Output: Dictionary
#               * e.g.
#                    
#               {'2024-05-06': {'USD 10Y Real Rate': 0.0703732,
#                               'Copper': 1.10528,
#                                'VIX': 0.803888,
#                                'US 5Y Infl. Expec.': 0.0259345,
#                                'US 5s30s Swap': 0.0383366,
#                                'US 10Y Yield': 0.0609977,
#                                'WTI': 1.83589,
#                                'US HY': 0.0727548,
#                                'FED Rate Expectations': 0.046424,
#                                'FED QT Expectations': 0.0194275,
#                                'US GDP': 0.0981244,
#                                'USD TWI': 0.325864}
#                                 ...
#                                 }
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
try:
    api_response = api_instance.get_descriptor_stds_for_risk_model(risk_model, date_from=date_from, date_to=date_to)
    pprint(api_response)
except ApiException as e:
    print(f"Exception when calling DefaultApi:get_descriptor_stds_for_risk_model: {e}")

#############################################################################################################################################################################################
# 
# get_descriptor_stds_for_risk_model(risk_model, date_from=date_from, date_to=date_to)
# is a Qi Risk API endpoint to retrieve the standard deviations for factors within a risk model for a given time frame (max 1 year per query).
# 
# Inputs:
#               * date_from [date] (optional) - Start date of data required 'YYYY-MM-DD' format.
#               * date_to [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * risk_model[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
# 
# If your equity focus consists of approximately 90% US or Canadian assets, please use the US Model with or without the market. All securities have Global and US models.
#
# Output: Dictionary
# Units: %
#               * e.g.
#                    
#                       {'2023-01-02': {'Copper': 1.74467,
#                        'FED QT Expectations': 0.0302016,
#                        'FED Rate Expectations': 0.0612414,
#                        'US 10Y Yield': 0.0764862,
#                        'US 5Y Infl. Expec.': 0.0654564,
#                        'US 5s30s Swap': 0.054162,
#                        'US GDP': 0.124191,
#                        'US HY': 0.147182,
#                        'USD 10Y Real Rate': 0.0824326,
#                        'USD TWI': 0.517187,
#                        'VIX': 1.7345,
#                        'WTI': 3.06592},
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

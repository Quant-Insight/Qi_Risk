#############################################################################################################################################################################################
# 
# get_covariances_for_risk_model(risk_model, date_from=date_from, date_to=date_to)
# is a Qi Risk API endpoint to retrieve the covariances for the chosen risk model for a given time frame (max 1 year per query).
# 
# Inputs:
#               * date_from [date] (optional) - Start date of data required 'YYYY-MM-DD' format.
#               * date_to [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * risk_model[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#
#               If your equity focus consists of approximately 90% US or Canadian assets, please use the US Models with or without the market (QI_US_MACRO_MT_1). All securities have Global and US models.
#                                             
#
# Output: Dictionary
#               * e.g.
#                    
#               {'2024-05-06': {'CB QT Expectations': {'CB QT Expectations': 0.593011,
#                       'CB Rate Expectations': 0.254585,
#                       'Corporate Credit': 0.214164,
#                          'DM FX': 0.167989,
#                           'Economic Growth': 0.0417275,
#                          'Energy': 0.0150284,
#                           'Forward Growth Expectations': -0.135185,
#                           'Inflation': 0.212937,
#                           'Metals': -0.0039164,
#                           'Real Rates': 0.195478,
#                           'Risk Aversion': 0.119673,
#                           '10Y Yield': 0.364545},
#                          'CB Rate Expectations': {'CB QT Expectations': 0.254585,
#                           'CB Rate Expectations': 0.590145,
#                           'Corporate Credit': 0.057664,
#                          'DM FX': 0.0969394,
#                           'Economic Growth': -0.118534,
#                           'Energy': 0.0364193,
#                           'Forward Growth Expectations': -0.346582,
#                           'Inflation': 0.259192,
#                           'Metals': 0.0437236,
#                           'Real Rates': 0.0141556,
#                           'Risk Aversion': -0.0133477,
#                           '10Y Yield': 0.584879}
#                        ...
#                          }
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

from pprint import pprint


risk_model = 'QI_US_MACRO_MT_1' 
date_from = '2023-01-01'  
date_to = '2023-12-31' 

try:
    # Call to the get_covariances_for_risk_model endpoint
    api_response = api_instance.get_covariances_for_risk_model(risk_model, date_from=date_from, date_to=date_to)
    pprint(api_response)
except ApiException as e:
    print(f"Exception when calling DefaultApi:get_covariances_for_risk_model: {e}")

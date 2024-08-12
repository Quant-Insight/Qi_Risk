#############################################################################################################################################################################################
#
# get_factor_returns_for_risk_model(risk_model, date_from=date_from, date_to=date_to)
# is a QI API endpoint to retrieve the factor returns for a risk model for a given time frame (max 1 year per query).
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
#                     {'2023-01-02': {'10Y Yield': 0.48414,
#                       'CB QT Expectations': 0.35296,
#                       'CB Rate Expectations': -0.0224032,
#                       'Corporate Credit': 0.0,
#                       'DM FX': 0.324678,
#                       'Economic Growth': -0.0089379,
#                       'Energy': 0.0,
#                       'Forward Growth Expectations': -0.421661,
#                       'Inflation': -0.0381933,
#                       'Metals': 0.0,
#                       'Real Rates': 0.0092197,
#                       'Risk Aversion': 0.0},
#                             ...
#                             }  
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
    api_response = api_instance.get_factor_returns_for_risk_model(risk_model, date_from = date_from, date_to = date_to)
    pprint(api_response)
except ApiException as e:
    print(f"Exception when calling DefaultApi:get_descriptor_stds_for_risk_model: {e}")

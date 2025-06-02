#############################################################################################################################################################################################
# 
# get_factor_returns_stds_for_risk_model(risk_model, date_from=date_from, date_to=date_to)
# is a Qi Risk API endpoint to retrieve the factor returns stds for the chosen risk model for a given time frame (max 1 year per query).
# 
# Inputs:
#               * date_from [date] (optional) - Start date of data required 'YYYY-MM-DD' format.
#               * date_to [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * risk_model[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#
# If your equity focus consists of approximately 90% US or Canadian assets, please use the US Model with or without the market. All securities have Global models.
#                                             
#
# Output: Dictionary
#               * e.g.
#                    
#               {'2023-01-02': {'10Y Yield': 0.0763353,
#                  'CB QT Expectations': 0.0301548,
#                  'CB Rate Expectations': 0.0611474,
#                  'Corporate Credit': 0.146888,
#                  'DM FX': 0.516155,
#                 'Economic Growth': 0.123943,
#                  'Energy': 3.05978,
#                 'Forward Growth Expectations': 0.0540564,
#                  'Inflation': 0.0653313,
#                  'Metals': 1.74118,
#                  'Real Rates': 0.0822678,
#                  'Risk Aversion': 1.73103},
#               '2023-01-03': {'10Y Yield': 0.0763353,
#                  'CB QT Expectations': 0.0302113,
#                  'CB Rate Expectations': 0.0612867,
#                  'Corporate Credit': 0.146966,
#                  'DM FX': 0.518227,
#                  'Economic Growth': 0.123942,
#                  'Energy': 3.07113,
#                  'Forward Growth Expectations': 0.0541178,
#                  'Inflation': 0.0653254,
#                  'Metals': 1.74231,
#                  'Real Rates': 0.0824977,
#                  'Risk Aversion': 1.73274},
#               '2023-01-04': {'10Y Yield': 0.0771335,
#                ...
#                  'Inflation': 0.0376373,
#                  'Metals': 1.21178,
#                  'Real Rates': 0.0785453,
#                  'Risk Aversion': 0.972089}}
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
    api_response = api_instance.get_factor_returns_stds_for_risk_model(risk_model, date_from=date_from, date_to=date_to)
    pprint(api_response)
except ApiException as e:
    print(f"Exception when calling DefaultApi:get_covariances_for_risk_model: {e}")

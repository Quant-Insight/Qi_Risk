#############################################################################################################################################################################################
#
# get_exposures_for_risk_model(risk_model, identifier , date_from=date_from,date_to=date_to)
# is a QI API endpoint to retrieve the exposures for the factor set for a defined identifier during a given time frame.
# 
# Inputs:
#               * date_from [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * date_to [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * risk_model[str] (required) - The Factor Set e.g 'QI_US_MACRO_MT_1'
#               * identifier[str] (required) - The indentifier that you want to pull data from
# Output: Dictionary
#               * e.g.
#                    
#                   {'2023-01-02': {'10Y Yield': 0.0347415,
#                     'CB QT Expectations': 0.117077,
#                     'CB Rate Expectations': 0.0402197,
#                     'Corporate Credit': -0.58322,
#                     'DM FX': -0.135774,
#                     'Economic Growth': -0.0347323,
#                     'Energy': 0.0251908,
#                     'Forward Growth Expectations': 0.13837,
#                     'Inflation': 0.0419626,
#                     'Metals': 0.0047402,
#                     'Real Rates': -0.0397995,
#                     'Risk Aversion': -0.381538},
#                      ...
#                            }
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
identifier='SPX Index' 

try:
    api_response = api_instance.get_exposures_for_risk_model(risk_model, identifier = identifier, date_from=date_from, date_to=date_to)
    pprint(api_response)
  
except ApiException as e:
    print(f"Exception when calling DefaultApi:get_descriptor_stds_for_risk_model: {e}")


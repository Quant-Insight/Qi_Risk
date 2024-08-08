#############################################################################################################################################################################################
#
# get_risk_model_factorset(risk_model)
# is a QI API endpoint to retrieve risk model's factorset.
# 
# Inputs:
#               * risk_model[str] (required) - The Factor Set e.g 'QI_US_MACRO_MT_1'
#
# Output: Dictionary
#       [{'descriptor': 'USD 10Y Real Rate', 'factor': 'Real Rates'},
#       {'descriptor': 'Copper', 'factor': 'Metals'},
#       {'descriptor': 'VIX', 'factor': 'Risk Aversion'},
#       {'descriptor': 'US 5Y Infl. Expec.', 'factor': 'Inflation'},
#       {'descriptor': 'US 5s30s Swap', 'factor': 'Forward Growth Expectations'},
#       {'descriptor': 'US 10Y Yield', 'factor': '10Y Yield'},
#       {'descriptor': 'WTI', 'factor': 'Energy'},
#       {'descriptor': 'US HY', 'factor': 'Corporate Credit'},
#       {'descriptor': 'FED Rate Expectations', 'factor': 'CB Rate Expectations'},
#       {'descriptor': 'FED QT Expectations', 'factor': 'CB QT Expectations'},
#       {'descriptor': 'US GDP', 'factor': 'Economic Growth'},
#       {'descriptor': 'USD TWI', 'factor': 'DM FX'}]
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

try:
    api_response = api_instance.get_risk_model_factorset(risk_model)
    pprint(api_response)
except ApiException as e:
    print(f"Exception when calling DefaultApi:get_descriptor_stds_for_risk_model: {e}")
#############################################################################################################################################################################################
#
# get_risk_model_factorset(risk_model)
# is a QI API endpoint to retrieve a risk model's factorset.
# 
# Inputs:
#               * risk_model[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#  
# If your equity focus consists of approximately 90% US or Canadian assets, please use the US Model with or without the market. All securities have Global and US models.
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
    print(f"Exception when calling DefaultApi:get_risk_model_factorset: {e}")

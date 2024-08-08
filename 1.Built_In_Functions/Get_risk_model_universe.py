#############################################################################################################################################################################################
#
# get_risk_model_universe(risk_model, identifier_type=identifier_type,include_delisted=include_delisted)
# is a QI API endpoint to retrieve risk model data for a specific instrument during a given time frame.
# 
# Inputs:
#               * risk_model[str] (required) - The Factor Set. If your equity focus consists of approximately 90% US or Canadian assets, please use the US Models with or without the market (QI_US_MACRO_MT_1). All securities have Global and US models.
#               * identifier_type[str] (Optional) -Type of identifier to return the universe - the default is SEDOL (optional)
#               * include_delisted[str](Optional)- Flag to include delisted instruments (True/False)
# 
# Output: List
#               * e.g.
#                    
#                ['B07DRZ5',
#                 'BMQ5W17',
#                 'B0DJNG0',
#                 'B14NJ71',
#                 'BYZ1856',
#                 'B3DGH82',
#                 '2490847',
#                 'BZ9NWS6',
#                 'BMWCNP3',
#                 'B3KFWW1',
#                 'BKT4TC7',
#                 'BFFJPG7',
#                 'B61TVQ0',
#                 '0709954',
#                 '2232878',
#                 'BBX4LR9',
#                 'B50P5B5',
#                 'B572ZV9'
#                  ...
#                 ]
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
# Type of identifier to return the universe - the default is SEDOL (optional)
# Datatype: str
identifier_type = 'SEDOL'
# Flag to include delisted instruments (optional)
# Datatype: bool
include_delisted = True

try:
    api_response = api_instance.get_risk_model_universe(risk_model, identifier_type=identifier_type, include_delisted=include_delisted)
    pprint(api_response)
  
except ApiException as e:
    print(f"Exception when calling DefaultApi:get_risk_model_universe: {e}")

#############################################################################################################################################################################################
#
# get_risk_models_for_identifier()
# is a QI API endpoint to retrieve all of the risk models.
# Inputs:
#                * identifier: identifier used to find risk models for which it is a constituent of (e.g. AAPL US Equity).
# Output: List
#
#               ['QI_EU_MACRO_MT_1',
#                'QI_US_MACRO_MT_1',
#                'QI_EU_MACROMKT_MT_1',
#                'QI_US_MACROMKT_MT_1',
#                'QI_GLOBAL_MACROMKT_MT_1',
#                'QI_GLOBAL_MACRO_MT_1']
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

# identifier used to find risk models for which it is a constituent of.
# Datatype: str
identifier = 'AAPL US Equity'

try:
    api_response = api_instance.get_risk_models_for_identifier(identifier)
    pprint(api_response)

except ApiException as e:
    print(f"Exception when calling DefaultApi:get_risk_models_for_identifier: {e}")

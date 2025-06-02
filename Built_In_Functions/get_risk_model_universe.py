#############################################################################################################################################################################################
#
# get_risk_model_universe(risk_model, identifier_type=None, include_delisted=False, tags=None, asset_classes=None)
# is a QI API endpoint to retrieve all instruments available for a specific risk model.
# 
# Inputs:
#               * risk_model[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#               * identifier_type[str] (optional) - The identifier you want risk models for. Identifier can be a SEDOL, ISIN or Bloomberg Ticker. 
#                                                   If not specified, it will return all the identifiers available for each instrument.
#               * include_delisted[bool] (optional) - boolean type paramater for including delisted models (True or False), default is False.
#               * tags[str] (optional) - comma delimited list of tags to filter results with (e.g. 'S&P 1500').
#               * asset_class[str] (optional) - comma delimited list of asset classes to filter results with. Results must contain all asset
#                                          classes specified (e.g. 'Equity').
# Output: List
#               * e.g.
#                    {'A': {'SEDOL': '2520153'},
#                     'AAPL': {'SEDOL': '2046251'},
#                     'ABBV': {'SEDOL': 'B92SR70'},
#                     'ABNB': {'SEDOL': 'BMGYYH4'},
#                     'ABT': {'SEDOL': '2002305'},
#                     ...
#                     'YUM': {'SEDOL': '2098876'},
#                     'ZBH': {'SEDOL': '2783815'},
#                     'ZBRA': {'SEDOL': '2989356'},
#                     'ZTS': {'SEDOL': 'B95WG16'}}
#                
#############################################################################################################################################################################################

import qi_client
from qi_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: Qi API Key
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = 'YOUR-API-KEY'

# Uncomment to use proxy - please refer to Connectivity Guidance doc for more details.
# configuration.proxy = 'http://corporateproxy.business.com:8080'

# Instantiate API class.
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

risk_model = 'QI_US_MACROMKT_MT_1'
identifier_type = 'SEDOL'
include_delisted = False
tags = 'S&P 500'
asset_classes = 'Equity'

try:
    api_response = api_instance.get_risk_model_universe(risk_model, 
                                                        identifier_type=identifier_type, 
                                                        include_delisted=include_delisted,
                                                        tags=tags, 
                                                        asset_classes=asset_classes
                                                        )
    pprint(api_response)

except ApiException as e:
    print(f"Exception when calling DefaultApi:get_risk_model_universe: {e}")

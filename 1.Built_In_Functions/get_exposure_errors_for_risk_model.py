#############################################################################################################################################################################################
#
# get_exposure_errors_for_risk_model(risk_model, instrument/identifier, date_from=date_from, date_to=date_to)
# is a QI API endpoint to retrieve the exposures errors for a given identifier and risk model across a period of time (max 1 year per query).
# 
# Inputs:
#               * date_from [date] (optional) - Start date of data required 'YYYY-MM-DD' format.
#               * date_to [date] (optional) - End date of data required 'YYYY-MM-DD' format.
#               * risk_model[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#               * instrument[str] or identifier[str] (required) - The instrument or identifier you want to pull data for. For instrument use Qi's default asset naming convention, e.g. 'AAPL'.
#                                                                 For identifier use asset's SEDOL, ISIN or Bloomberg Ticker.
#
# If your equity focus consists of approximately 90% US or Canadian assets, please use the US Model with or without the market. All securities have Global and US models.
#
# Output: Dictionary
# Units: %
#               * e.g.
#                    
#                {'2023-01-02': {'10Y Yield': 0.0155746,
#                              'CB QT Expectations': 0.01628,
#                              'CB Rate Expectations': 0.0135608,
#                              'Corporate Credit': 0.018385,
#                              'DM FX': 0.0234206,
#                              'Economic Growth': 0.0128139,
#                              'Energy': 0.0073466,
#                              'Forward Growth Expectations': 0.0187662,
#                              'Inflation': 0.0061693,
#                              'Metals': 0.0118163,
#                              'Real Rates': 0.0085236,
#                              'Risk Aversion': 0.0169761},
#               '2023-01-03': {'10Y Yield': 0.0155458,
#                              'CB QT Expectations': 0.016554,
#                              'CB Rate Expectations': 0.0135226,
#                              'Corporate Credit': 0.0184887,
#                              'DM FX': 0.0234406,
#                              'Economic Growth': 0.0125814,
#                              'Energy': 0.0073454,
#                              'Forward Growth Expectations': 0.0186205,
#                              'Inflation': 0.0062148,
#                              'Metals': 0.011876,
#                              'Real Rates': 0.0086344,
#                              'Risk Aversion': 0.0167976},
#               '2023-01-04': {'10Y Yield': 0.0156951,
#              ...
#                              'Inflation': 0.0057023,
#                              'Metals': 0.0158217,
#                              'Real Rates': 0.0072011,
#                              'Risk Aversion': 0.017614}}
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
identifier = 'SPX Index' 

try:
    api_response = api_instance.get_exposure_errors_for_risk_model(risk_model, identifier=identifier, date_from=date_from, date_to=date_to)
    pprint(api_response)
except ApiException as e:
    print(f"Exception when calling DefaultApi:get_exposure_errors_for_risk_model: {e}")

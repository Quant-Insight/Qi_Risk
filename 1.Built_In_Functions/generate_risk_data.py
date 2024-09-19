#############################################################################################################################################################################################
#
# generate_risk_data(returns, risk_model)
# This function uses the Qi API function run_risk_model(model, timeseries) to generate the risk model data, given an asset's returns timeseries. 
# 
# Inputs: 
#               * returns[dict] (required) - A dictionary with the returns timeseries of the asset you want to run the risk model for. 
#                  * e.g. 
#                    {'2019-01-02': 0.26554799904132675,
#                     '2019-01-03': -3.4559658043694985,
#                     '2019-01-04': 3.855906026542577,
#                     '2019-01-07': 0.8018853643327217,
#                     '2019-01-08': 1.5917950633580702,
#                     '2019-01-09': 1.474110276785101,
#                     '2019-01-10': 0.42476188704061446,
#                     '2019-01-11': -0.053172361586673045,
#                     '2019-01-14': 0.0963514745678129,
#                     '2019-01-15': -0.7629296185424739,
#                     '2019-01-16': 0.48423254458196396,
#                     '2019-01-17': 1.5595063070390047,
#                     '2019-01-18': 2.5964266377910272,
#                      ...
#               * risk_model[str] (required) - 8 models (4 regions with and without market): QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, QI_EU_MACRO_MT_1,
#                                                                                            QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
# Output: dict - For each day within the output, you can access the following data:
#                * factor_returns
#                * covariance_matrix
#                * specific_returns
#                * exposures
#                * descriptor_returns_stds 
#
#               * e.g.
#                {'2019-12-18': {'covariance_matrix': {'10Y Yield': {'10Y Yield': 1.49775,
#                                                                    'CB QT Expectations': -0.34904,
#                                                                    'CB Rate Expectations': 0.58622,
#                                                                    'Corporate Credit': -0.43649,
#                                                                    'DM FX': 0.23205,
#                                                                    'Economic Growth': 0.10844,
#                                                                    'Energy': 0.25305,
#                                                                    'Forward Growth Expectations': 0.14672,
#                                                                    'Inflation': 0.66417,
#                                                                    'Metals': 0.47383,
#                                                                    'Real Rates': 1.21673,
#                                                                    'Risk Aversion': -0.3682},
#                                                      'CB QT Expectations': {'10Y Yield': -0.34904,
#                                                                             'CB QT Expectations': 0.88901,
#                                                                             'CB Rate Expectations': -0.1652,
#                                                                             'Corporate Credit': 0.26928,
#                                                                             'DM FX': -0.07494,
#                                                      ...
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

def process_risk_response(response):
        risk_data = response['results']
        last_evaluated_key = response.get('last_evaluated_key', None)
        return risk_data, last_evaluated_key

def generate_risk_data(returns, risk_model):
        
        job = api_instance.run_risk_model(risk_model, {"timeseries": returns})
        job_id = job['job_id']

        # Wait for job to enter cache
        sleep(5)

        # Check job status
        job_status = api_instance.check_risk_job_status(job_id)
        status = job_status['algo']

        # Wait until job is complete
        while status != 'COMPLETE':
            sleep(10)
            job_status = api_instance.check_risk_job_status(job_id)
            status = job_status['algo']

        try:
            # Get list of all defined models on the system  
            response = api_instance.get_risk_model_data(job_id)

            risk_data, exclusive_start_key = process_risk_response(response)

            while exclusive_start_key:
                
                response = api_instance.get_risk_model_data(
                    job_id,
                    exclusive_start_key=exclusive_start_key
                )
                _risk_data, exclusive_start_key = process_risk_response(response)
                risk_data.update(_risk_data)
        
        except ApiException as e:
            print("Exception when calling DefaultApi->get_models_with_pagination: %s\n" % e)

        return(risk_data)


returns = {...}
risk_model = 'QI_US_MACRO_MT_1'

try:
  risk_data = generate_risk_data(returns, risk_model)
  pprint(risk_data)
except ApiException as e:
  print(f"Exception when calling DefaultApi:generate_risk_data: {e}")

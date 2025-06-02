# Qi Risk

This repository includes some examples of how to use Quant-Insight's API. 

It contains three main folders:

  * Built_In_Functions: contains Qi Risk API's built-in functions.
  * Portfolio: Script to analyse your portfolio.
  * Use_Cases: Ready-to-use examples of our API. 

## What do you need to start using the API?

* Client Download Token

  * This will be unique to your organisation and is required in order to install the Qi Client. 

* API key

  * If you still don't have an API key, you can contact Quant-Insight. 
  
  * If you already have an API key:
          
          * Install QI client, with your TOKEN instead of DLTOKEN (Note that to install packages on 
          Jupyter Notebooks you need to use !pip install instead of pip install):

                !pip install matplotlib pandas

                !pip install \
                --upgrade \
                --extra-index-url=https://dl.cloudsmith.io/DLTOKEN/quant-insight/python/python/index/ \
                qi-client
               
           * Insert the following piece of code at the start of your script, with your API key instead 
           of 'ADD-YOUR-API-KEY-HERE': 

                import pandas
                import qi_client
                from qi_client.rest import ApiException

                configuration = qi_client.Configuration()

                configuration.api_key['X-API-KEY'] = 'ADD-YOUR-API-KEY-HERE'

                api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

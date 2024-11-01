### Code
  
    #########################################################################################################
    # 
    # This function calculates an instrument's exposures (i.e. SPY) to one of our risk models 
    # (i.e. QI_US_MACROMKT_MT_1) for a chosen date.
    #
    # Requirements:
    #               import pandas
    #               import Qi_risk_wrapper
    #               import numpy as np
    #               from datetime import datetime, timedelta
    #               from dateutil.relativedelta import relativedelta
    #
    # Inputs: 
    #               * date [str] (required) - Date of data required 'YYYY-MM-DD' format.
    #               * asset [str] (required) - Asset to calculate the exposures for.
    #               * risk_model[str] (required) - 8 models (4 regions with and without market):
    #                            QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, 
    #                            QI_EU_MACRO_MT_1, QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, 
    #                            QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
    #
    # 
    # Output: 
    #               * DataFrame - Instrument's exposures to each descriptor within the chosen model. 
    #
    #                            10Y Yield  CB QT Expectations  CB Rate Expectations  Corporate Credit  ...  
    #                2024-10-28  -0.050193            0.068423              0.037172         -0.410114  ... 
    #
    #
    #########################################################################################################
    
    
    
    from Qi_risk_wrapper import RiskData
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    
    if __name__ == "__main__":
    
        model = 'QI_US_MACRO_MT_1'
        asset = 'SPY'
        date = '2024-10-28'
        # api_key = 'YOUR-API-KEY'
    
        try:
    
            risk_data = RiskData(model, asset, date, date, api_key)
            exposures_df = risk_data.exposures
            print(exposures_df)
    
        except Exception as e:
            print(f'Unexpected error: {e}')

### Visualization ideas
After pulling the data for your asset (i.e. SPY), you can create a bar chart showing a snapshot of an instrument's exposures:

![factor_exposure_bar_chart_SPY](https://github.com/user-attachments/assets/0e599ac2-2421-4cc5-b8db-bd9d2c4c0666)


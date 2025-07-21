### Visualization ideas

After pulling the data for your asset (i.e. SPY) using the code below, you can create a scatter plot with the actual and 
factor return values for the selected period. 

![return_attribution_scatter_historical_SPY](https://github.com/user-attachments/assets/c7211bcd-2cc3-4cb6-8dbd-53ddba20796b)


### Code

    #########################################################################################################
    # 
    # This function calculates an instrument's actual and factor returns (i.e. SPY) to one of our risk models 
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
    #               * date_from [str] (required) - Start date of data required 'YYYY-MM-DD' format.
    #               * date_to [str] (required) - End date of data required 'YYYY-MM-DD' format.
    #               * asset [str] (required) - Asset to calculate the factor and total returns for.
    #               * risk_model[str] (required) - 8 models (4 regions with and without market):
    #                            QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, 
    #                            QI_EU_MACRO_MT_1, QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, 
    #                            QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
    #
    # 
    # Output: 
    #               * DataFrame - Factor and Actual return attribution using the predifined periods. 
    #
    #                            total_return  factor_return
    #                2023-10-27     -0.453278      -0.305265
    #                2023-10-30      1.195580       0.960202
    #                2023-10-31      0.628023       0.654028
    #                2023-11-01      1.066480       1.247370
    #                2023-11-02      1.916430       1.982990
    #                ...                  ...            ...
    #                2024-10-23     -0.913735      -0.584665
    #                2024-10-22     -0.053116      -0.028829
    #                2024-10-24      0.216267       0.290473
    #                2024-10-25     -0.034528      -0.394823
    #                2024-10-28      0.309132       0.446574
    #
    #               * float - Slope of the line of best fit.
    #                
    #                 Slope: 0.8848486631113682
    #
    #               * float - Intercept of the line of best fit.
    #                 
    #                 Intercept: -0-0.09154904457773186
    #           
    #               * float - R² value of the line of best fit.
    #
    #                 R²: 0.7085538552967738  
    #
    #########################################################################################################
  
    
    from datetime import datetime, timedelta
    from typing import Tuple
    
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression  # type: ignore
    
    from Qi_risk_wrapper import RiskData
    
    
    def get_actual_vs_factor_daily_returns(
        risk_data: RiskData,
    ) -> Tuple[pd.DataFrame, float, float, float]:
    
        risk_model_data_df = risk_data.risk_model_data[
            ['total_return', 'factor_return']
        ]
    
        # Calculate the line of best fit
        x_data = np.array(risk_model_data_df['total_return'].tolist())
        y_data = np.array(risk_model_data_df['factor_return'].tolist())
    
        x_data_reshaped = x_data.reshape(-1, 1)
        model = LinearRegression().fit(x_data_reshaped, y_data)
        y_fit = model.predict(x_data_reshaped)
    
        # Get the equation of the line of best fit
        slope = model.coef_[0]
        intercept = model.intercept_
    
        # Calculate the R² value
        r_squared = model.score(x_data_reshaped, y_data)
    
        return risk_model_data_df, slope, intercept, r_squared
    
    
    if __name__ == "__main__":
    
        model = 'QI_US_MACRO_MT_1'
        asset = 'SPY'
        date_from = '2023-10-27'
        date_to = '2024-10-28'
        # api_key = 'YOUR-API-KEY'
    
        try:
    
            risk_data = RiskData(model, asset, date_from, date_to, api_key)
    
            risk_model_data_df, slope, intercept, r_squared = (
                get_actual_vs_factor_daily_returns(risk_data)
            )
    
            print(risk_model_data_df)
            print(f'Slope: {slope}')
            print(f'Intercept: {intercept}')
            print(f'R²: {r_squared}')
    
        except Exception as e:
            print(f'Unexpected error: {e}')



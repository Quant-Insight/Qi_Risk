### Code

#################################################################################################
# 
# This code takes a portfolio and creates an Excel file with its exposures, factor attribution 
# and risk proportion and contribution. 
#
# Requirements:
#               import Qi_risk_portfolio_wrapper
#               import pandas as pd
#               import os
#               import subprocess
#               import sys
#               import joblib
#               import matplotlib
#               import openpyxl
#
# Inputs: 
#               * DIR [str] (required) - Path to your portfolio. 
#               * portfolio_name [str] required - Name of the csv (.csv) file containing the 
#                                                 portfolio to analyse.
#                                                 
#                                                 The file must have the following format:
#
#                                                 | Identifier     | Weight |
#                                                 | 2046251        | 0.3    |
#                                                 | ...            | ...    |
#
#               * portfolio_analysis_name [str] (required) - Name of the file with the output data 
#                                                            of the portfolio analysis.
#               * identifier_type [str] (required) - Identifier type included in the portfolio.
#                                                    It can take the following values: SEDOL
#                                                    
# 
#               * date_from [str] (required) - Start date of data required 'YYYY-MM-DD' format.
#               * date_to [str] (required) - End date of data required 'YYYY-MM-DD' format.
#               * model[str] (required) - 8 models (4 regions with and without market):
#                            QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, 
#                            QI_EU_MACRO_MT_1, QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, 
#                            QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#
# 
# Output: 		
#        * Excel file showing the missing assets.
#        * Excel file with the assets without enough historical data.
#        * Excel file with the covered portfolio analysis, including the following sheets:


from typing import List

import pandas as pd
import os
import subprocess
import sys

os.environ['QI_API_KEY'] = 'YOUR-API-KEY'
    
from portfolio_wrapper import PortfolioRiskData
from portfolio_wrapper import ApiData
from portfolio_wrapper import RiskData
from portfolio_wrapper import PortfolioAnalysis
from portfolio_wrapper import PorfolioRiskExcel

DIR = '/Users/aidapendasfernandez/Documents/Github/PC_Proff/Risk/portfolio_analysis'

if __name__ == '__main__':

    portfolio_analysis_name = 'portfolio_output'
    portfolio_name = 'test_portfolio.csv'
    model = 'QI_US_MACRO_MT_1'
    date_from = '2025-01-02'
    date_to = '2025-07-18'
    identifier_type = 'SEDOL'

    api_data = ApiData()
    portfolioExcel = PorfolioRiskExcel()
    portfolio_data = PortfolioRiskData(model)

    # Get portfolio's coverage
    risk_model_universe = api_data.get_universe_by_model(
        model, identifier_type=identifier_type, include_delisted=True
    )

    df_portfolio = pd.read_csv(DIR + '/' + portfolio_name)

    missing_instruments, missing_data, instruments = ( 
        portfolio_data.get_portfolio_coverage(
            df_portfolio,
            risk_model_universe,
            model,
            date_from,
            date_to,
            identifier_type,
        )
    )

    df_portfolio[
        df_portfolio['Identifier'].isin(missing_instruments)
    ].to_excel(DIR + '/' + portfolio_analysis_name + '_missing.xlsx')

    if len(missing_data) > 0:
        df_portfolio[df_portfolio['Identifier'].isin(missing_data)].to_excel(
            DIR + '/' + portfolio_analysis_name + '_missing_data.xlsx'
        )

    weights = df_portfolio[
        ~df_portfolio['Identifier'].isin(missing_instruments + missing_data)
    ]['Weight'].tolist()

    portfolioExcel.portfolio_risk_to_excel(
        DIR,
        portfolio_analysis_name,
        model,
        instruments,
        weights,
        date_from,
        date_to,
    )

    file_path_excel = f'{DIR}/{portfolio_analysis_name}_portfolio_{model}_{date_to}.xlsx'

    data = PortfolioAnalysis(file_path_excel)
    data.runner()

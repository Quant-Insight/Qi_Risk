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
    #
    # Inputs: 
    #               * DIR [str] (required) - Path to your portfolio. 
    #               * portfolio_name [str] required - Name of the Excel(.xlsx) file containing the 
    #                                                 portfolio to analyse.
    #                                                 
    #                                                 The file must have the following format:
    #
    #                                                 | Identifier     | Weight |
    #                                                 | AAPL US Equity | 0.3    |
    #                                                 | ...            | ...    |
    #
    #               * portfolio_analysis_name [str] (required) - Name of the file with the output data 
    #                                                            of the portfolio analysis.
    #               * identifier_type [str] (required) - Identifier type included in the portfolio.
    #                                                    It can take the following values:
    #                                                    
    #                                                    BloombergTicker, ISIN or SEDOL
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
    #        * Excel file with the covered portfolio analysis, including the following sheets:



                  
![image](https://github.com/user-attachments/assets/770c3aed-9616-4de6-88ee-f3703da8b7da)




    
    from typing import List
    
    import pandas as pd
    import os
    import subprocess
    import sys
        
    os.environ['QI_API_KEY'] = 'YOUR-API-KEY'
    
    from Qi_risk_portfolio_wrapper import PortfolioRiskData
    from Qi_risk_portfolio_wrapper import ApiData
    from Qi_risk_portfolio_wrapper import RiskData
    
    DIR = 'PATH-TO-YOUR-PORTFOLIO'
    
    def portfolio_risk_to_excel(
        name: str,
        model: str,
        assets: List[str],
        weights: List[str],
        date_from: str,
        date_to: str,
    ) -> None:
    
        # Initialise portfolio risk data class and pull/calculate required data.
        portfolio_risk = PortfolioRiskData(model)
        portfolio_risk.get_data(assets, weights, date_from, date_to)
    
        factor_attribution = portfolio_risk.get_factor_attribution()
        factor_risk_proportion = portfolio_risk.get_factor_proportion_of_risk()
        factor_contribution_to_risk = (
            portfolio_risk.get_factor_contribution_to_risk(annualised=False)
        )
        stock_proportion_of_risk, stock_contribution_to_risk = (
            portfolio_risk.get_portfolio_risk_ts_by_stock(annualised=False)
        )
        risk_by_stock = portfolio_risk.get_factor_risk_by_stock(
            date_to, annualised=False
        )
        factor_attribution_by_stock = (
            portfolio_risk.get_factor_attribution_by_stock_for_period(
                lookback=3 * 22
            )
        )
        exposures_by_stock = portfolio_risk.get_weighted_stock_exposures_for_date(
            date_to
        )
    
        file_path = f'{DIR}/{name}_portfolio_{model}_{date_to}.xlsx'
    
        # Create a Pandas Excel writer using Openpyxl as the engine
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Write data to different sheets
            portfolio_risk.exposures.to_excel(
                writer, sheet_name='exposures', index=True
            )
            factor_attribution.to_excel(
                writer, sheet_name='factor_attribution', index=True
            )
            portfolio_risk.risk_model_data.to_excel(
                writer, sheet_name='risk_and_return_ts', index=True
            )
            factor_risk_proportion.to_excel(
                writer, sheet_name='factor_risk_proportion', index=True
            )
            factor_contribution_to_risk.to_excel(
                writer, sheet_name='factor_risk_contribution', index=True
            )
            exposures_by_stock.to_excel(
                writer, sheet_name=f'exposures_{date_to}', index=True
            )
            factor_attribution_by_stock.to_excel(
                writer, sheet_name='factor_attribution_3m', index=True
            )
            risk_by_stock.to_excel(
                writer, sheet_name=f'stock_risk_{date_to}', index=True
            )
            stock_proportion_of_risk.to_excel(
                writer, sheet_name='stock_risk_proportion', index=True
            )
            stock_contribution_to_risk.to_excel(
                writer, sheet_name='stock_risk_contribution', index=True
            )
    
    
    if __name__ == '__main__':
    
        portfolio_analysis_name = 'PORTFOLIO-OUTPUT_FILE_NAME'
        portfolio_name = 'PORTFOLIO-FILE-NAME.xlsx'
        model = 'QI_US_MACRO_MT_1'
        date_from = '2024-01-01'
        date_to = '2024-10-25'
        identifier_type = 'BloombergTicker' # 'BloombergTicker', 'ISIN' or 'SEDOL'
    
        # Get portfolio's coverage
        api_data = ApiData()
    
        portfolio_data = PortfolioRiskData(model)
    
        risk_model_universe = api_data.get_universe_by_model(
            model, identifier_type=identifier_type, include_delisted=False
        )
    
        df_portfolio = pd.read_excel(DIR + '/' + portfolio_name)
    
        missing_instruments, missing_data, instruments = (
            portfolio_data.get_portfolio_coverage(
                df_portfolio, risk_model_universe, model, date_from, date_to
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
    
        portfolio_risk_to_excel(
            portfolio_analysis_name,
            model,
            instruments,
            weights,
            date_from,
            date_to,
        )

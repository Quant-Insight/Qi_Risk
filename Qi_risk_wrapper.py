#######################################################################################################################################
# 
# This module allows you to call all Qi use-case functions. 
#
# Download this script and save in your working directory. 
# Set your API Key as an environment variable.
# Import Qi_risk_wrapper
#
# Requirements:
#
# For Jupyter Notebooks use (API Key needs to be declared without ''):
#
#         import Qi_risk_wrapper
#
# For other Python development environments (API Key needs to be declared as a string ''):
#
#         import Qi_risk_wrapper 
#
#######################################################################################################################################


from typing import Callable, Dict, List

import numpy as np
import pandas as pd

import qi_client
from qi_client.rest import ApiException

RISK_WIN = 125
HALF_LIFE = 90

class ApiData:
    """
    Class that provides methods to retrieve data using Qi's API.
    """

    def __init__(self, api_key: str):
        """
        Initializes the ApiData class by assigning the user's API key.

        Args:
            api_key (str): User's API key provided by Qi.
        """

        configuration = qi_client.Configuration() 

        configuration.api_key['X-API-KEY'] = api_key
        self.api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))


    @staticmethod
    def _get_data_for_func(
        func: Callable,
        model: str,
        start: str,
        end: str,
        asset: str = None,
    ) -> Dict:
        """
        Helper method to retrieve data for a given function and time period, splitting the data
        by year to handle large date ranges.

        Args:
            func (Callable): The API function to be called for data retrieval.
            model (str): The risk model used.
            start (str): The start date in 'YYYY-MM-DD' format.
            end (str): The end date in 'YYYY-MM-DD' format.
            asset (str, optional): The asset for which data is being retrieved. Default is None.

        Returns:
            Dict: The retrieved data as a dictionary.
        """
        year_start = int(start[:4])
        year_end = int(end[:4])
        data = {}

        for year in range(year_start, year_end + 1):

            if year != year_start:
                date_from = '%d-01-01' % year
            else:
                date_from = start

            if year != year_end:
                date_to = '%d-12-31' % year
            else:
                date_to = end

            if asset:
                data.update(
                    func(
                        model,
                        instrument=asset,
                        date_from=date_from,
                        date_to=date_to,
                    )
                )

            else:
                data.update(
                    func(
                        model,
                        date_from=date_from,
                        date_to=date_to,
                    )
                )

        return data

    def get_exposure_data(
        self, model: str, asset: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        func = self.api_instance.get_exposures_for_risk_model
        exposures = self._get_data_for_func(
            func, model, date_from, date_to, asset
        )

        return pd.DataFrame.from_dict(exposures).T

    def get_risk_model_data(
        self, model: str, asset: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        func = self.api_instance.get_risk_model_data_for_risk_model
        risk_model_data = self._get_data_for_func(
            func, model, date_from, date_to, asset
        )

        return pd.DataFrame.from_dict(risk_model_data).T

    def get_factor_returns_data(
        self, model: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        func = self.api_instance.get_factor_returns_for_risk_model
        factor_returns = self._get_data_for_func(
            func, model, date_from, date_to
        )

        return pd.DataFrame.from_dict(factor_returns).T

    def get_factor_stds_data(
        self, model: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        func = self.api_instance.get_descriptor_stds_for_risk_model
        factor_stds = self._get_data_for_func(func, model, date_from, date_to)

        return pd.DataFrame.from_dict(factor_stds).T

    def get_factor_covariance_data(
        self, model: str, date_from: str, date_to: str
    ) -> Dict:
        func = self.api_instance.get_covariances_for_risk_model
        cov_data = self._get_data_for_func(func, model, date_from, date_to)

        return cov_data
    

class RiskData:
    """
    This class gets risk data for an asset via Qi's API, and has several methods
    to derive further risk metrics (return attribtuions and vol/risk predictions).

    Attributes:
        exposures (pd.DataFrame): Asset exposures to various factors.
        exposure_errors (pd.DataFrame): Errors associated with asset exposures.
        risk_model_data (pd.DataFrame): Risk model data for the asset over the specified period.
        factor_returns (pd.DataFrame): Factor returns for the specified time period.
        factor_stds (pd.DataFrame): Rolling standard deviations of risk factors over time.
        factor_covariance (Dict): Covariance data for risk factors over time.
    """

    def __init__(self, model: str, asset: str, date_from: str, date_to: str, api_key: str):
        """
        Initializes the RiskData class by retrieving necessary risk data for a given asset
        over the specified time range.

        Args:
            model (str): The risk model to be used for data retrieval.
            asset (str): The asset for which risk data is being retrieved.
            date_from (str): The start date for the data in 'YYYY-MM-DD' format.
            date_to (str): The end date for the data in 'YYYY-MM-DD' format.
            api_key (str): User's API key provided by Qi.
        """
        self.api_data = ApiData(api_key)
        self.exposures = self.api_data.get_exposure_data(
            model, asset, date_from, date_to
        )
        self.risk_model_data = self.api_data.get_risk_model_data(
            model, asset, date_from, date_to
        )
        self.factor_returns = self.api_data.get_factor_returns_data(
            model, date_from, date_to
        )
        self.factor_stds = self.api_data.get_factor_stds_data(
            model, date_from, date_to
        )
        self.factor_covariance = self.api_data.get_factor_covariance_data(
            model, date_from, date_to
        )

    def get_factor_attribution(self) -> pd.DataFrame:
        """
        Calculates factor attributions, which represent the contributions of each factor to the asset's returns.
        For factor return attributions we use exposures on t-1, and factor returns on t, to calculate factor
        attributions on t.

        Returns:
            pd.DataFrame: A DataFrame containing factor attributions.
        """
        factor_attribution = self.exposures.shift(1) * self.factor_returns
        return factor_attribution

    def get_cumulative_attribution(self) -> pd.DataFrame:
        """
        Calculates the cumulative return attribution, given an asset.

        Returns:
            pd.DataFrame: Cumulative return attribution table.
        """    
        risk_model_data_df = self.risk_model_data[['total_return', 'factor_return']].rename(columns = {'total_return': 'actual', 'factor_return': 'factor'})

        attribution_df = 100 * ((((risk_model_data_df / 100) + 1).cumprod()) - 1)
        attribution_df['specific'] = (attribution_df['actual'] - attribution_df['factor'])

        return attribution_df

    @staticmethod
    def get_weighted_returns(
        returns: pd.DataFrame, half_life: int
    ) -> pd.DataFrame:
        """
        Computes weighted returns using an exponentially decaying weighting scheme.

        Args:
            returns (pd.DataFrame): A DataFrame containing returns data.
            half_life (int): The half-life period for decay in the weighting scheme.

        Returns:
            pd.DataFrame: A DataFrame containing the weighted returns.
        """
        window = len(returns)
        expo_weights = [
            (0.5 ** (1 / half_life)) ** (window - x)
            for x in range(1, window + 1)
        ]
        expo_weights = [x / np.mean(expo_weights) for x in expo_weights]

        weighted_returns = returns.mul(expo_weights, axis=0)

        return weighted_returns

    def get_factor_risk(self, annualised: bool = False) -> pd.DataFrame:
        """
        Calculates the factor risk for each date, optionally annualized.

        Total_risk = (factor_risk**2 + specific_risk**2)**0.5
        factor_risk**2 = X.F.X  ,
        where X is exposures, F is covariance matrix
        specific_risk**2 = variance of exponentially weighted specific returns

        Args:
            annualised (bool, optional): Whether to annualize the risk values. Default is False.

        Returns:
            pd.DataFrame: A DataFrame containing factor risk values for each date.
        """
        exposures = self.exposures
        specific = self.risk_model_data.specific_return
        factor_results = []

        if len(specific) < RISK_WIN:
            raise Exception(
                f'{RISK_WIN} days required for factor risk calcs, {len(specific)} days provided'
            )

        for date in specific.index[RISK_WIN:]:
            date_exposures = exposures.loc[date]
            date_specific = self.get_weighted_returns(
                specific[:date][-RISK_WIN:], HALF_LIFE
            ).var()
            date_covar_matrix = pd.DataFrame(self.factor_covariance[date])
            date_exposures = date_exposures[date_covar_matrix.index]

            factor_risk = np.dot(date_exposures, date_covar_matrix)
            factor_risk_df = (
                pd.DataFrame(factor_risk, index=date_exposures.index).T
                * date_exposures
            )

            scalar = 252**0.5 if annualised else 1

            factor_result = {'date': date}
            for factor, value in factor_risk_df.loc[0].items():
                if value < 0:
                    value = -1 * (abs(value) ** 0.5) * scalar
                else:
                    value = (value**0.5) * scalar
                factor_result.update({factor: value})
            factor_results.append(factor_result)

        factor_risk_df = pd.DataFrame(factor_results).set_index('date')
        return factor_risk_df

    def get_factor_proportion_of_risk(self) -> pd.DataFrame:
        """
        Calculates the proportion of risk attributable to each factor (and specific).

        Returns:
            pd.DataFrame: A DataFrame containing the proportion of risk by factor.
        """
        factor_risk = self.get_factor_risk()
        total_risk = self.risk_model_data['total_risk']
        factor_risk_proportion = (
            (factor_risk**2 * np.sign(factor_risk))
            .div(total_risk**2, axis=0)
            .dropna()
        )
        factor_risk_proportion['specific'] = (
            self.risk_model_data['specific_risk'] ** 2
        ) / (self.risk_model_data['total_risk'] ** 2)

        return factor_risk_proportion

    def get_factor_contribution_to_risk(
        self, annualised: bool = False
    ) -> pd.DataFrame:
        """
        Calculates the contribution of each factor (and specific) to the overall risk of the asset.

        Args:
            annualised (bool, optional): Whether to annualize the risk contributions. Default is False.

        Returns:
            pd.DataFrame: A DataFrame containing the factor contributions to risk.
        """
        factor_risk_proportion = self.get_factor_proportion_of_risk()
        total_risk = self.risk_model_data['total_risk']
        factor_contribution_to_risk = factor_risk_proportion.mul(
            total_risk, axis=0
        ).dropna()
        return factor_contribution_to_risk

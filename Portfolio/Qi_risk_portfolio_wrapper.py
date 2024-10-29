from typing import Callable, Dict, List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import qi_client
from qi_client.rest import ApiException

import os

QI_API_KEY = os.environ.get('QI_API_KEY', None)
if not QI_API_KEY:
    print("Mandatory environment variable QI_API_KEY not set!")

configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = QI_API_KEY
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

RISK_WIN = 125
HALF_LIFE = 90
ZERO_IDX = 0


class RiskData(ABC):
    """
    Abstract base class for retrieving risk data for an asset via Qi's API, and has
    several methods to derive further risk metrics (return attribtuions and vol/risk predictions).

    Attributes:
        model (str): The risk model used to retrieve risk data.
        exposures (pd.DataFrame): Asset exposures to various risk factors.
        exposure_errors (pd.DataFrame): Errors associated with asset exposures.
        risk_model_data (pd.DataFrame): Risk model data for the asset.
        factor_returns (pd.DataFrame): Factor returns data for the factors in the risk model.
        factor_stds (pd.DataFrame): Rolling standard deviations for risk factors.
        factor_risk (pd.DataFrame): Factor risk values for the asset.
        factor_covariance (Dict[str, Dict[str, Dict[str, float]]]): Covariance matrix for factors.
    """

    def __init__(self, model: str):
        self.model = model
        self.api_data = ApiData()
        self.exposures: pd.DataFrame = pd.DataFrame()
        self.exposure_errors: pd.DataFrame = pd.DataFrame()
        self.risk_model_data: pd.DataFrame = pd.DataFrame()
        self.factor_returns: pd.DataFrame = pd.DataFrame()
        self.factor_stds: pd.DataFrame = pd.DataFrame()
        self.factor_risk: pd.DataFrame = pd.DataFrame()
        self.factor_covariance: Dict[str, Dict[str, Dict[str, float]]] = {}

    @abstractmethod
    def get_data(self, **kwags) -> None:
        """
        Abstract method to retrieve data for the asset.

        This method must be implemented by subclasses to define the specific data retrieval
        logic based on the asset type, model, and other parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for data retrieval, which may vary based on the subclass.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_risk(self) -> pd.DataFrame:
        """
        Abstract method to calculate the risk for an asset.

        Subclasses must implement this method to provide calculations for risk forecasts
        based on the asset and model.

        Returns:
            pd.DataFrame: DataFrame containing factor risk values.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError()

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

    @staticmethod
    def _get_proportion_of_risk(
        risk: [pd.DataFrame, pd.Series], total_risk: pd.Series
    ) -> pd.DataFrame:
        """
        Helper function to calculate the proportion of risk attributable to each factor (and specific).

        Args:
            risk [pd.DataFrame, pd.Series]: Either a DataFrame or Series of risk values.
            total_risk: A data series of total risk values.

        Returns:
            pd.DataFrame: A DataFrame containing the proportion of risk by factor.
        """
        risk = pd.DataFrame(risk)
        risk_proportion = (
            (risk**2 * np.sign(risk)).div(total_risk**2, axis=0).dropna()
        )

        return risk_proportion

    def get_factor_proportion_of_risk(self) -> pd.DataFrame:
        """
        Calculates the proportion of risk attributable to each factor (and specific).

        Returns:
            pd.DataFrame: A DataFrame containing the proportion of risk by factor.
        """
        factor_risk_proportion = self._get_proportion_of_risk(
            self.factor_risk, self.risk_model_data.total_risk
        )

        specific_risk_proportion = self._get_proportion_of_risk(
            self.risk_model_data.specific_risk, self.risk_model_data.total_risk
        )
        factor_risk_proportion['specific'] = specific_risk_proportion[
            'specific_risk'
        ]

        return factor_risk_proportion

    @staticmethod
    def _get_contribution_to_risk(
        risk_proportion: pd.DataFrame,
        total_risk: pd.Series,
        annualised: bool = False,
    ) -> pd.DataFrame:
        """
        Helper function to calculate the contribution of each factor (and specific) to the overall risk of the asset.

        Args:
            risk_proportion (pd.DataFrame): DataFrame of factor risk proportions.
            total_risk (pd.Series): data Series of total risk values.
            annualised (bool, optional): Whether to annualize the risk contributions. Default is False.

        Returns:
            pd.DataFrame: A DataFrame containing the factor contributions to risk.
        """
        scalar = 252**0.5 if annualised else 1
        contribution_to_risk = risk_proportion.mul(
            total_risk * scalar, axis=0
        ).dropna()

        return contribution_to_risk

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
        factor_contribution_to_risk = self._get_contribution_to_risk(
            factor_risk_proportion, self.risk_model_data.total_risk, annualised
        )

        return factor_contribution_to_risk


class AssetRiskData(RiskData):

    def get_data(self, asset: str, date_from: str, date_to: str):
        """
        Retrieves risk data for a given asset over the specified time range.

        Args:
            asset (str): The asset for which risk data is being retrieved.
            date_from (str): The start date for the data in 'YYYY-MM-DD' format.
            date_to (str): The end date for the data in 'YYYY-MM-DD' format.
        """
        self.exposures = self.api_data.get_exposure_data(
            self.model, asset, date_from, date_to
        )
        self.exposure_errors = self.api_data.get_exposure_error_data(
            self.model, asset, date_from, date_to
        )
        self.risk_model_data = self.api_data.get_risk_model_data(
            self.model, asset, date_from, date_to
        )
        self.factor_returns = self.api_data.get_factor_returns_data(
            self.model, date_from, date_to
        )
        self.factor_stds = self.api_data.get_factor_stds_data(
            self.model, date_from, date_to
        )
        self.factor_covariance = self.api_data.get_factor_covariance_data(
            self.model, date_from, date_to
        )
        print(f'Fetched data for {asset}, in model {self.model}')

        self.factor_risk = self.calculate_risk()

    def calculate_risk(self, annualised: bool = False) -> pd.DataFrame:
        """
        Calculates the factor risk for each date, optionally annualized.

        Total_risk = (factor_risk**2 + specific_risk**2)**0.5
        factor_risk**2 = X.F.X  ,
        where X is exposures, F is covariance matrix

        Args:
            annualised (bool, optional): Whether to annualize the risk values. Default is False.

        Returns:
            pd.DataFrame: A DataFrame containing factor risk values for each date.
        """
        factor_results = []

        for date in self.exposures.index:
            date_exposures = self.exposures.loc[date]
            date_covar_matrix = pd.DataFrame(self.factor_covariance[date])
            date_exposures = date_exposures[date_covar_matrix.index]
            factor_risks = (
                date_exposures.dot(date_covar_matrix)
            ) * date_exposures

            scalar = 252**0.5 if annualised else 1

            factor_result = {'date': date}
            for factor, value in factor_risks.items():
                if value < 0:
                    value = -1 * (abs(value) ** 0.5) * scalar
                else:
                    value = (value**0.5) * scalar
                factor_result.update({factor: value})
            factor_results.append(factor_result)

        factor_risk_df = pd.DataFrame(factor_results).set_index('date')

        return factor_risk_df


class ApiData:
    """
    Class that provides methods to retrieve data using Qi's API.
    """

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
        func = api_instance.get_exposures_for_risk_model
        exposures = self._get_data_for_func(
            func, model, date_from, date_to, asset
        )

        return pd.DataFrame.from_dict(exposures).T

    def get_exposure_error_data(
        self, model: str, asset: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        func = api_instance.get_exposure_errors_for_risk_model
        exposure_errors = self._get_data_for_func(
            func, model, date_from, date_to, asset
        )

        return pd.DataFrame.from_dict(exposure_errors).T

    def get_risk_model_data(
        self, model: str, asset: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        func = api_instance.get_risk_model_data_for_risk_model
        risk_model_data = self._get_data_for_func(
            func, model, date_from, date_to, asset
        )

        return pd.DataFrame.from_dict(risk_model_data).T

    def get_factor_returns_data(
        self, model: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        func = api_instance.get_factor_returns_for_risk_model
        factor_returns = self._get_data_for_func(
            func, model, date_from, date_to
        )

        return pd.DataFrame.from_dict(factor_returns).T

    def get_factor_stds_data(
        self, model: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        func = api_instance.get_descriptor_stds_for_risk_model
        factor_stds = self._get_data_for_func(func, model, date_from, date_to)

        return pd.DataFrame.from_dict(factor_stds).T

    def get_factor_covariance_data(
        self, model: str, date_from: str, date_to: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        func = api_instance.get_covariances_for_risk_model
        cov_data = self._get_data_for_func(func, model, date_from, date_to)

        return cov_data

    def get_universe_by_model(
        self, risk_model, identifier_type, include_delisted
    ):
        api_response = api_instance.get_risk_model_universe(
            risk_model,
            identifier_type=identifier_type,
            include_delisted=include_delisted,
        )
        return api_response


class PortfolioRiskData(RiskData):
    """
    A subclass of RiskData that calculates and manages portfolio risk data for a
    portfolio of assets. This class retrieves risk data for a portfolio, calculates
    exposures, risk, and returns on a portfolio level, and provides methods to analyze
    risk contributions by factor and asset.
    """

    def get_data(
        self,
        assets: List[str],
        weights: List[float],
        date_from: str,
        date_to: str,
    ) -> None:
        """
        Retrieves risk data for a given list of assets and weights
        in a portfolio over the specified time range.

        Args:
            assets List[str]: list of assets in portfolio.
            weights List[float]: list of portfolio weights.
            date_from (str): The start date for the data in 'YYYY-MM-DD' format.
            date_to (str): The end date for the data in 'YYYY-MM-DD' format.

        Raises:
            AssertionError: If the number of assets does not match the number of weights.

        Returns:
            None: The method retrieves and stores risk data internally for further calculations.
        """
        assert len(weights) == len(
            assets
        ), 'array of assets must be of same length as weights'
        self.weights = weights
        self.asset_data: Dict[str, AssetRiskData] = {}
        for asset in assets:
            asset_risk_data = AssetRiskData(self.model)
            asset_risk_data.get_data(asset, date_from, date_to)
            self.asset_data[asset] = asset_risk_data

        self.exposures = self.calculate_portfolio_exposures()

        self.factor_returns = self.api_data.get_factor_returns_data(
            self.model, date_from, date_to
        )
        self.factor_stds = self.api_data.get_factor_stds_data(
            self.model, date_from, date_to
        )
        self.factor_covariance = self.api_data.get_factor_covariance_data(
            self.model, date_from, date_to
        )
        self.exposures = self.exposures.dropna()
        self.factor_risk = self.calculate_risk()

    def calculate_portfolio_exposures(self) -> pd.DataFrame:
        """
        Calculates weighted exposures for each factor based on
        individual asset exposures and portfolio weights.

        Returns:
            pd.DataFrame: A DataFrame containing the weighted portfolio exposures for each factor.
        """

        weighted_exposures = sum(
            [
                x.exposures * self.weights[i]
                for i, x in enumerate(self.asset_data.values())
            ]
        )

        return weighted_exposures

    def calculate_risk(self) -> pd.DataFrame:
        """
        Calculates the factor risk and risk_model_data (total, specific and factor risk and return)
        for each date, optionally annualized (just applies for risk values).

        Total_risk = (factor_risk**2 + specific_risk**2)**0.5
        factor_risk**2 = X . F . X  ,
        where X is weighted portfolio exposures, F is covariance matrix
        specific_risk**2 = w.T . R . w
        where w is vector of portfolio weights, R is diagonal matrox of specific risk variances.

        Returns:
            pd.DataFrame: DataFrame that contains the factor risk values for each date.
        """

        weights_df = pd.DataFrame(self.weights, index=self.asset_data.keys())
        portfolio_factor_returns = self.get_factor_attribution().T.sum()

        factor_results = []
        aggregate_results = []
        for date in self.exposures.index:
            date_exposures = self.exposures.loc[date]
            date_covar_matrix = pd.DataFrame(self.factor_covariance[date])
            factor_risks = (
                self.exposures.loc[date].dot(date_covar_matrix)
            ) * self.exposures.loc[date]

            diag_specific_risk_matrix = np.diag(
                [
                    x.risk_model_data.specific_risk[date] ** 2
                    for x in self.asset_data.values()
                ]
            )
            weighted_diag_specific_risk_matrix = np.dot(
                np.dot(weights_df.T, diag_specific_risk_matrix), weights_df
            )

            factor_var = factor_risks.sum()
            specific_var = weighted_diag_specific_risk_matrix[ZERO_IDX][
                ZERO_IDX
            ]

            factor_risk = factor_var**0.5
            specific_risk = specific_var**0.5
            total_risk = (factor_var + specific_var) ** 0.5

            total_return = sum(
                [
                    x.risk_model_data.total_return[date] * self.weights[i]
                    for i, x in enumerate(self.asset_data.values())
                ]
            )
            factor_return = portfolio_factor_returns[date]
            specific_return = total_return - factor_return

            aggregate_results.append(
                {
                    'date': date,
                    'total_return': total_return,
                    'factor_return': factor_return,
                    'specific_return': specific_return,
                    'total_risk': total_risk,
                    'factor_risk': factor_risk,
                    'specific_risk': specific_risk,
                }
            )

            factor_result = {'date': date}
            for factor, value in factor_risks.items():
                if value < 0:
                    value = -1 * (abs(value) ** 0.5)
                else:
                    value = value**0.5
                factor_result.update({factor: value})
            factor_results.append(factor_result)

        factor_risk_df = pd.DataFrame(factor_results).set_index('date')
        risk_df = pd.DataFrame(aggregate_results).set_index('date')

        self.risk_model_data = risk_df

        return factor_risk_df

    def get_portfolio_risk_ts_by_stock(
        self, annualised: bool = False
    ) -> [pd.DataFrame, pd.DataFrame]:
        """
        Calculates the time series of portfolio risk due to each stock in the portfolio,
        split into factor and specific risks from each stock.

        Args:
            annualised (bool, optional): Whether to annualize the risk values. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - The first DataFrame contains the proportion of portfolio risk attributable to each stock.
                - The second DataFrame contains each stock's contribution to the total portfolio risk.
        """

        weights_df = pd.DataFrame(self.weights, index=self.asset_data.keys())

        stock_results = []
        for date in self.exposures.index:
            date_exposures = pd.DataFrame(
                [x.exposures.loc[date] for x in self.asset_data.values()],
                index=self.asset_data.keys(),
            )
            date_covar_matrix = pd.DataFrame(self.factor_covariance[date])
            stock_factor_risk = (date_exposures.dot(date_covar_matrix)).dot(
                date_exposures.T
            )
            weighted_stock_factor_risk = (
                weights_df.T.dot(stock_factor_risk) * weights_df.T
            )

            diag_specific_var_matrix = np.diag(
                [
                    x.risk_model_data.specific_risk[date] ** 2
                    for x in self.asset_data.values()
                ]
            )
            weighted_stock_specific_risk = (
                np.dot(weights_df.T, diag_specific_var_matrix) * weights_df.T
            )

            day_result = {'date': date}
            for stock, value in weighted_stock_factor_risk.loc[
                ZERO_IDX
            ].items():
                if value < 0:
                    value = -1 * (abs(value) ** 0.5)
                else:
                    value = value**0.5
                day_result.update({f'{stock}_factor': value})

            for stock, value in weighted_stock_specific_risk.loc[
                ZERO_IDX
            ].items():
                day_result.update({f'{stock}_specific': value**0.5})

            stock_results.append(day_result)

        stock_risks = pd.DataFrame(stock_results).set_index('date')

        stock_proportion_of_risk = self._get_proportion_of_risk(
            stock_risks, self.risk_model_data.total_risk
        )
        stock_contribution_to_risk = self._get_contribution_to_risk(
            stock_proportion_of_risk,
            self.risk_model_data.total_risk,
            annualised,
        )

        return stock_proportion_of_risk, stock_contribution_to_risk

    def get_factor_risk_by_stock(
        self, date: str, annualised: bool = False
    ) -> pd.DataFrame:
        """
        Retrieves the factor risk contribution for each stock in the portfolio for a specific date.
        These are estimates for breaking down risk into each factor for each stock.

        Args:
            date (str): The specific date for which to retrieve factor risk data.
            annualised (bool, optional): Whether to annualize the risk values. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing factor risk values for each stock on the specified date.
        """
        results = {}
        for asset, weight in zip(self.asset_data, self.weights):
            asset_risk = self.asset_data[asset].factor_risk.loc[date] * weight
            asset_risk *= sqrt(252) if annualised else 1
            results[asset] = asset_risk.to_dict()

        return pd.DataFrame.from_dict(results).T

    def get_factor_attribution_by_stock_for_period(
        self, lookback: int = 22
    ) -> pd.DataFrame:
        """
        Calculates the return attribution from each factor for each stock in the portfolio over a lookback period,
        which represents the contribution of each factor to the weighted stock's return over the period.

        Args:
            lookback (int, optional): The number of days to look back for the attribution calculation. Defaults to 22.

        Returns:
            pd.DataFrame: A DataFrame containing the factor return attribution for each stock over the specified period.
        """
        result = {}
        for asset, weight in zip(self.asset_data, self.weights):
            factor_attribution = (
                self.asset_data[asset].get_factor_attribution()[-lookback:]
                * weight
            )
            period_attribution = (
                (1 + (factor_attribution / 100)).prod() - 1
            ) * 100
            result[asset] = period_attribution.to_dict()

        return pd.DataFrame.from_dict(result).T

    def get_weighted_stock_exposures_for_date(self, date: str) -> pd.DataFrame:
        """
        Retrieves the weighted factor exposures for each stock in the portfolio on a specific date.

        Args:
            date (str): The date for which to retrieve weighted exposures.

        Returns:
            pd.DataFrame: A DataFrame containing the weighted exposures for each stock in the portfolio on the given date.
        """
        result = pd.DataFrame(
            [
                x.exposures.loc[date] * w
                for x, w in zip(self.asset_data.values(), self.weights)
            ],
            index=self.asset_data.keys(),
        )
        return result

    def flatten_dict(self, d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_portfolio_coverage(
        self,
        df_portfolio,
        risk_model_universe,
        risk_model,
        date_from,
        date_to,
    ):

        portfolio_universe = df_portfolio['Identifier'].astype(str)

        portfolio_mapping = api_instance.get_instruments_from_identifiers(
            {
                "identifiers": portfolio_universe.tolist(),
                "target_date": date_to,
            }
        )
        portfolio_identifiers = list(
            portfolio_mapping['resolved_instruments'].values()
        )

        missing_identifiers = portfolio_mapping['unresolved_identifiers']

        risk_model_mapping = api_instance.get_instruments_from_identifiers(
            {
                "identifiers": risk_model_universe,
                "target_date": date_to,
            }
        )
        model_identifiers = list(
            risk_model_mapping['resolved_instruments'].values()
        )

        existing_identifiers = [
            instrument
            for instrument in portfolio_identifiers
            if instrument not in missing_identifiers
        ]

        df_portfolio['Instrument'] = existing_identifiers

        api_data = ApiData()
        missing_historical_data = [
            df_portfolio[df_portfolio.Instrument == instrument][
                'Identifier'
            ].tolist()[0]
            for instrument in existing_identifiers
            if len(
                api_data.get_exposure_data(
                    risk_model, instrument, date_from, date_to
                )
            )
            == 0
        ]

        covered_identifiers = [
            instrument
            for instrument in existing_identifiers
            if instrument
            not in df_portfolio[
                df_portfolio.Identifier.isin(missing_historical_data)
            ]['Instrument'].tolist()
        ]

        return (
            missing_identifiers,
            missing_historical_data,
            covered_identifiers,
        )

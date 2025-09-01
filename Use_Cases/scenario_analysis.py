#########################################################################################################
# 
# This function builds a one-day “scenario P&L” for a list of assets given a list of core factors and a 
# risk model (i.e. QI_US_MACROMKT_MT_1).
#
# Requirements:
#               import pandas
#               import numpy as np
#               from datetime import datetime, timedelta
#               from dateutil.relativedelta import relativedelta
#               from retrying import retry
#
# Inputs: 
#               * analysis_date [str] (required) - Analysis date for data. Required 'YYYY-MM-DD' format.
#               * asset_list [list] (required) - Assets to calculate the scenario analysis for.
#               * risk_model[str] (required) - 8 models (4 regions with and without market):
#                            QI_US_MACROMKT_MT_1, QI_US_MACRO_MT_1, QI_EU_MACROMKT_MT_1, 
#                            QI_EU_MACRO_MT_1, QI_APAC_MACROMKT_MT_1, QI_APAC_MACRO_MT_1, 
#                            QI_GLOBAL_MACROMKT_MT_1, QI_GLOBAL_MACRO_MT_1.
#               * core_factors = [list] (required) - List of factors
#               * shocks_percent = [list] (required) - Input shocks as percentages
#
# 
# Output: 
#               * DataFrame - Scenario analysis for all the specified assets. 
#
#                                                                     AAPL	      META	      MSFT
#                Peripheral Impact (CB QT Expectations)	            0.063827	 -0.759291	   0.078839
#                Peripheral Impact (CB Rate Expectations)	          0.093767	  1.224307	  -0.058702
#                Peripheral Impact (DM FX)	                        0.020340	 -0.029754	  -0.034401
#                Peripheral Impact (Economic Growth)	              0.217659	 -0.189605	   0.069028
#                Peripheral Impact (Energy)	                       -0.109202    0.070733	  -0.010401
#                Peripheral Impact (Forward Growth Expectations)	 -0.269165	 -0.001072	  -0.090072
#                Peripheral Impact (Metals)	                        0.029646	  0.060739	   0.036157
#                Peripheral Impact (Real Rates)	                    0.037301	 -0.013151	  -0.038902
#                Peripheral Impact (Risk Aversion)	               -2.472159	 -2.452713	  -2.797113
#                Peripheral Impact (10Y Yield)	                    1.001371	 -0.775926	   0.375929
#                Total Peripheral Impact	                         -1.386615	 -2.865733	  -2.469638
#                Direct Impact (Corporate Credit)	                 -1.910101	 -3.344635	  -2.803243
#                Direct Impact (Inflation)	                       -3.068826	 -2.459342	  -2.427318
#                Total Direct Impact	                             -4.978927	 -5.803977	  -5.230561
#                Total Impact	                                     -6.365543	 -8.669709	  -7.700199
#
#########################################################################################################


import pandas as pd
import qi_client
from qi_client.rest import ApiException
import numpy as np
from retrying import retry

configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = 'YOUR-API-KEY' 

api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

# ==============================================================================
# CONFIGURATION AND SETUP
# ==============================================================================

# Input assumptions
risk_model = 'QI_US_MACRO_MT_1'
analysis_date = '2025-01-28'  # Single date for the stress test
# Define assets to process
asset_list = ['AAPL', 'META', 'MSFT']

# Define factor shocks in PERCENTAGE terms
# Option 1: List format (must match order of core_factors)
core_factors = ["Corporate Credit", "Inflation"]
shocks_percent = [0.5, 0.5]  # Input shocks as percentages



# ==============================================================================
# DATA RETRIEVAL
# ==============================================================================

# Calculate historical date range (125 business days back)
date_to_dt = pd.to_datetime(analysis_date)
date_from = (date_to_dt - pd.tseries.offsets.BDay(125)).strftime('%Y-%m-%d')

# Retrieve factor standard deviations (for converting % shocks to STDs)
factor_stds = api_instance.get_factor_returns_stds_for_risk_model(
    risk_model, 
    date_from=analysis_date, 
    date_to=analysis_date
)
factor_stds_df = pd.DataFrame(factor_stds).T

# Convert percentage shocks to standard deviation units
shocks_in_stds = []  # Store the converted shocks
for factor, shock_pct in zip(core_factors, shocks_percent):
    factor_std = factor_stds_df[factor].iloc[0]
    shock_std = shock_pct / factor_std
    shocks_in_stds.append(shock_std)
    

# Define the shocks in standard deviation units
r2_shocks = pd.Series(
    shocks_in_stds,
    index=core_factors
)

# Retrieve covariance matrix (for the analysis date)
cov_data = api_instance.get_covariances_for_risk_model(
    risk_model, 
    date_from=analysis_date, 
    date_to=analysis_date
)

# Handle the covariance data structure
if isinstance(cov_data, dict):
    if analysis_date in cov_data:
        cov_matrix = pd.DataFrame(cov_data[analysis_date])
    else:
        # Get the first (and likely only) date's data
        first_key = list(cov_data.keys())[0]
        cov_matrix = pd.DataFrame(cov_data[first_key])
else:
    cov_matrix = pd.DataFrame(cov_data)


# Retrieve historical factor returns (for 125-day mean calculation)
factor_returns = api_instance.get_factor_returns_for_risk_model(
    risk_model, 
    date_from=date_from, 
    date_to=analysis_date
)

# Handle the factor returns data structure
if isinstance(factor_returns, dict):
    sample_keys = list(factor_returns.keys())[:5] if factor_returns else []
    if sample_keys and all(isinstance(k, str) and '-' in k for k in sample_keys):
        factor_returns_df = pd.DataFrame(factor_returns).T
    else:
        factor_returns_df = pd.DataFrame(factor_returns)
else:
    factor_returns_df = pd.DataFrame(factor_returns)

# Ensure proper orientation (dates as index, factors as columns)
if factor_returns_df.shape[0] < factor_returns_df.shape[1]:
    factor_returns_df = factor_returns_df.T

factor_returns_df = factor_returns_df.sort_index(ascending=True)

# ==============================================================================
# COMPUTE HISTORICAL MEANS
# ==============================================================================

# Compute the 125-day rolling mean for each factor
mu_125_df = factor_returns_df.rolling(window=125).mean().iloc[-1].to_frame().T

# Extract means for core and peripheral factors
mu_2 = mu_125_df[core_factors]  # Historical means of core factors
mu_1 = mu_125_df.drop(columns=core_factors)  # Historical means of peripheral factors


# ==============================================================================
# EXTRACT COVARIANCE SUB-MATRICES
# ==============================================================================

# Extract covariance sub-matrices
Sigma_12 = cov_matrix.loc[mu_1.columns, mu_2.columns]  # Peripheral-Core covariances
Sigma_22 = cov_matrix.loc[mu_2.columns, mu_2.columns]  # Core-Core covariances

# Compute the inverse of Sigma_22
Sigma_22_inv = np.linalg.inv(Sigma_22)

# Compute Beta matrix
beta_matrix = Sigma_12 @ Sigma_22_inv
beta_matrix.columns = core_factors

# ==============================================================================
# PREPARE SHOCK ADJUSTMENTS
# ==============================================================================

# Ensure mu_2 is always a Series
if isinstance(mu_2, (float, np.float64)):
    mu_2 = pd.Series([mu_2], index=r2_shocks.index)
elif isinstance(mu_2, pd.DataFrame):
    mu_2 = mu_2.squeeze()

# Ensure alignment of index names
if isinstance(mu_2, pd.Series):
    mu_2.index.name = None

# Compute mean-adjusted shocks
r2_adjusted = r2_shocks - mu_2

# Compute time_shock (expected factor returns given the shock)
time_shock = beta_matrix @ r2_adjusted.T

# Convert mu_1 from DataFrame to Series
mu_1_series = mu_1.iloc[0]

# Align factor names
time_shock = time_shock.reindex(mu_1_series.index)

# Compute expected returns given r2
expected_r1_given_r2 = time_shock + mu_1_series

# ==============================================================================
# PROCESS ASSETS AND COMPUTE IMPACTS
# ==============================================================================

# Initialize results list
results = []

for asset in asset_list:
    
    try:
        # Retrieve exposures for the asset
        exposures = api_instance.get_exposures_for_risk_model(
            risk_model, 
            instrument=asset, 
            date_from=analysis_date, 
            date_to=analysis_date
        )
        exposures_df = pd.DataFrame.from_dict(exposures, orient="index")
        
        # Align factor names between expected_r1_given_r2 and exposures_df
        exposures_aligned = exposures_df[expected_r1_given_r2.index]
        
        # Compute peripheral impact contribution per factor
        impact_per_factor = expected_r1_given_r2 * exposures_aligned
        
        # Compute total peripheral impact
        total_peripheral_impact = impact_per_factor.sum(axis=1)
        
        # Extract exposures to core factors
        exposures_core = exposures_df[core_factors]
        
        # Compute direct impact per core factor
        direct_impact_per_core = exposures_core.iloc[0] * r2_shocks
        
        # Compute total direct impact
        total_direct_impact = direct_impact_per_core.sum()
        
        # Compute total impact
        total_impact = total_peripheral_impact + total_direct_impact
        
        # Store results
        result = {
            'Asset': asset,
            **{f'Peripheral Impact ({factor})': impact_per_factor[factor].iloc[0] 
               for factor in expected_r1_given_r2.index},
            'Total Peripheral Impact': total_peripheral_impact.iloc[0],
            **{f'Direct Impact ({factor})': direct_impact_per_core[factor] 
               for factor in core_factors},
            'Total Direct Impact': total_direct_impact,
            'Total Impact': total_impact.iloc[0]
        }
        
        results.append(result)
        
    except Exception as e:
        print(f"Skipping {asset} due to error: {e}")

results_df = pd.DataFrame(results).set_index("Asset").T
results_df.columns.name = None
results_df

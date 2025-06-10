"""
Data Generating Process for Two-Way Random Effects (TWRE and MTWRE) Models

This module implements the DGP for SFA with random effects in both frontier and variances.
- TWRE: Uncorrelated random effects 
- MTWRE: Mundlak-type correlated random effects
Based on the MATLAB codes DGP_TWRE.m and DGP_MTWRE.m by Kutlu and Sickles (2022).
"""

import numpy as np
import pandas as pd

def validate_dgp_output(Tdata, effTrue, dgp_name, ni, nt):
    """Validate DGP output and print basic statistics."""
    print(f"\n--- {dgp_name} DGP Validation ---")
    print(f"Data shape: {Tdata.shape}")
    print(f"Expected observations: {ni * nt}")
    print(f"Unique firms: {Tdata['id'].nunique()} (expected: {ni})")
    print(f"Unique time periods: {Tdata['ti'].nunique()} (expected: {nt})")
    print(f"Y statistics: mean={Tdata['y'].mean():.3f}, std={Tdata['y'].std():.3f}")
    print(f"True efficiency: mean={effTrue.mean():.3f}, std={effTrue.std():.3f}, min={effTrue.min():.3f}")
    print(f"Data columns: {list(Tdata.columns)}")

def dgp_twre(ni, nt, bf_true, bu_true, bv_true, het, validate=False):
    """
    Two-Way Random Effects DGP - Random effects in frontier and variances.
    alpha_i ~ N(0, het^2), gamma_u ~ N(0, het^2), gamma_v ~ N(0, het^2).
    All random effects are uncorrelated.
    
    Parameters:
    -----------
    ni : int
        Number of firms/cross-sectional units
    nt : int
        Number of time periods
    bf_true : array_like
        True frontier parameters [beta1, beta2, intercept]
    bu_true : array_like
        True inefficiency variance parameters
    bv_true : array_like
        True noise variance parameters
    het : float
        Heterogeneity parameter for random effects scale
    validate : bool
        Whether to print validation statistics
    
    Returns:
    --------
    Tdata : pandas.DataFrame
        Generated panel data with columns: y, xf1, xf2, xu, xv, id, ti
    effTrue : numpy.ndarray
        True efficiency values exp(-u)
    """
    nob = ni * nt
    
    # Generate regressors
    xf1 = np.random.randn(nob)
    xf2 = np.random.randn(nob)
    xu = np.random.randn(nob) - 1  # Used for inefficiency variance
    xv = np.random.randn(nob) - 2  # Used for noise variance
    
    # Panel structure
    id_ = np.repeat(np.arange(1, ni + 1), nt)
    ti = np.tile(np.arange(1, nt + 1), ni)
    
    # Random effects - all independent
    sigma_alpha = het
    sigma_gamma = het
    
    # Frontier random effect
    alpha_i = sigma_alpha * np.random.randn(ni)
    alpha_obs = alpha_i[id_ - 1]
    
    # Variance random effects
    gamma_u_i = sigma_gamma * np.random.randn(ni)
    gamma_v_i = sigma_gamma * np.random.randn(ni)
    gamma_u_obs = gamma_u_i[id_ - 1]
    gamma_v_obs = gamma_v_i[id_ - 1]
    
    # Variance components
    DMU = np.column_stack([xu, np.ones(nob)])  # Design matrix for bu
    DMV = np.column_stack([xv, np.ones(nob)])  # Design matrix for bv
    sigu = np.exp(DMU @ bu_true + gamma_u_obs)
    sigv = np.exp(DMV @ bv_true + gamma_v_obs)
    
    # Error terms
    u = sigu * np.abs(np.random.randn(nob))  # Half-normal inefficiency
    v = sigv * np.random.randn(nob)          # Normal noise
    
    # Output equation
    y = bf_true[2] + bf_true[0]*xf1 + bf_true[1]*xf2 + alpha_obs + v - u
    effTrue = np.exp(-u)
    
    # Create DataFrame
    Tdata = pd.DataFrame({
        'y': y, 'xf1': xf1, 'xf2': xf2, 'xu': xu, 'xv': xv,
        'id': id_, 'ti': ti
    })
    
    if validate:
        validate_dgp_output(Tdata, effTrue, 'TWRE', ni, nt)
    
    return Tdata, effTrue

def dgp_mtwre(ni, nt, bf_true, bu_true, bv_true, het, rho, validate=False):
    """
    Mundlak-type Two-Way Random Effects DGP.
    Random effects are correlated with firm-level means of regressors.
    
    Parameters:
    -----------
    ni : int
        Number of firms/cross-sectional units
    nt : int
        Number of time periods
    bf_true : array_like
        True frontier parameters [beta1, beta2, intercept]
    bu_true : array_like
        True inefficiency variance parameters
    bv_true : array_like
        True noise variance parameters
    het : float
        Heterogeneity parameter for random effects scale
    rho : float
        Correlation parameter for Mundlak-type correlation
    validate : bool
        Whether to print validation statistics
    
    Returns:
    --------
    Tdata : pandas.DataFrame
        Generated panel data with columns: y, xf1, xf2, xu, xv, id, ti
    effTrue : numpy.ndarray
        True efficiency values exp(-u)
    """
    nob = ni * nt
    
    # Generate regressors
    xf1 = np.random.randn(nob)
    xf2 = np.random.randn(nob)
    xu = np.random.randn(nob) - 1  # Used for inefficiency variance
    xv = np.random.randn(nob) - 2  # Used for noise variance
    
    # Panel structure
    id_ = np.repeat(np.arange(1, ni + 1), nt)
    ti = np.tile(np.arange(1, nt + 1), ni)
    
    # Scale parameters
    sigma_alpha = het
    sigma_gamma = het
    
    # Compute firm-level means using pandas groupby
    temp_df = pd.DataFrame({'id': id_, 'xf1': xf1, 'xu': xu, 'xv': xv})
    firm_means = temp_df.groupby('id').agg({'xf1': 'mean', 'xu': 'mean', 'xv': 'mean'}).reset_index()
    
    # Extract firm-level means in correct order
    mean_xf1 = firm_means['xf1'].values
    mean_xu = firm_means['xu'].values  
    mean_xv = firm_means['xv'].values
    
    # Alpha correlated with mean of xf1
    eps_alpha = np.random.randn(ni)
    alpha_i = sigma_alpha * (rho * mean_xf1 + np.sqrt(1 - rho**2) * eps_alpha)
    alpha_obs = alpha_i[id_ - 1]
    
    # Gamma_u correlated with mean of xu
    eps_u = np.random.randn(ni)
    gamma_u_i = sigma_gamma * (rho * mean_xu + np.sqrt(1 - rho**2) * eps_u)
    gamma_u_obs = gamma_u_i[id_ - 1]
    
    # Gamma_v correlated with mean of xv
    eps_v = np.random.randn(ni)
    gamma_v_i = sigma_gamma * (rho * mean_xv + np.sqrt(1 - rho**2) * eps_v)
    gamma_v_obs = gamma_v_i[id_ - 1]
    
    # Variance components
    DMU = np.column_stack([xu, np.ones(nob)])  # Design matrix for bu
    DMV = np.column_stack([xv, np.ones(nob)])  # Design matrix for bv
    sigu = np.exp(DMU @ bu_true + gamma_u_obs)
    sigv = np.exp(DMV @ bv_true + gamma_v_obs)
    
    # Error terms
    u = sigu * np.abs(np.random.randn(nob))  # Half-normal inefficiency
    v = sigv * np.random.randn(nob)          # Normal noise
    
    # Output equation
    y = bf_true[2] + bf_true[0]*xf1 + bf_true[1]*xf2 + alpha_obs + v - u
    effTrue = np.exp(-u)
    
    # Create DataFrame
    Tdata = pd.DataFrame({
        'y': y, 'xf1': xf1, 'xf2': xf2, 'xu': xu, 'xv': xv,
        'id': id_, 'ti': ti
    })
    
    if validate:
        validate_dgp_output(Tdata, effTrue, 'MTWRE', ni, nt)
    
    return Tdata, effTrue 
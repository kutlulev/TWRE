"""
Data Generating Process for True Random Effects (TRE) Stochastic Frontier Model

This module implements the DGP for SFA with random frontier effects.
Based on the MATLAB code DGP_TRE.m by Kutlu and Sickles (2022).
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

def dgp_tre(ni, nt, bf_true, bu_true, bv_true, het, validate=False):
    """
    True Random Effects DGP - Random firm effect in frontier only.
    alpha_i ~ N(0, het^2), gamma_u = gamma_v = 0.
    
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
        Heterogeneity parameter for random effects (sigma_alpha)
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
    
    # Random frontier effect: alpha_i ~ N(0, het^2)
    sigma_alpha = het
    alpha_i = sigma_alpha * np.random.randn(ni)
    alpha_obs = alpha_i[id_ - 1]  # Map to observations
    
    # No random effects in variances for TRE
    gamma_u_obs = np.zeros(nob)
    gamma_v_obs = np.zeros(nob)
    
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
        validate_dgp_output(Tdata, effTrue, 'TRE', ni, nt)
    
    return Tdata, effTrue 
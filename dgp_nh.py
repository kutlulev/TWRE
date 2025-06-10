"""
Data Generating Process for No Heterogeneity (NH) Stochastic Frontier Model

This module implements the DGP for standard SFA without random effects.
Based on the MATLAB code DGP_NH.m by Kutlu and Sickles (2022).
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

def dgp_nh(
    ni, nt,
    bf_true, bu_true, bv_true,
    s=1,
    validate=False
):
    """
    Generates data consistent with the NH.m MATLAB script.
    Model: y = X*beta + v + s*u
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
    
    # No random effects in NH model
    alpha_obs = np.zeros(nob)
    gamma_u_obs = np.zeros(nob) 
    gamma_v_obs = np.zeros(nob)
    
    # Variance components
    DMU = np.column_stack([xu, np.ones(nob)])  # Design matrix for bu
    DMV = np.column_stack([xv, np.ones(nob)])  # Design matrix for bv
    su_sq = np.exp(DMU @ bu_true + gamma_u_obs)
    sv_sq = np.exp(DMV @ bv_true + gamma_v_obs)
    
    # Error terms
    u = np.abs(np.random.standard_normal(nob)) * np.sqrt(su_sq)
    v = np.random.standard_normal(nob) * np.sqrt(sv_sq)
    
    # Output equation where s determines the sign of inefficiency
    y = bf_true[2] + bf_true[0]*xf1 + bf_true[1]*xf2 + alpha_obs + v + s * u
    effTrue = np.exp(-u)
    
    # Create DataFrame
    Tdata = pd.DataFrame({
        'y': y, 'xf1': xf1, 'xf2': xf2, 'xu': xu, 'xv': xv, 
        'id': id_, 'ti': ti
    })
    
    if validate:
        validate_dgp_output(Tdata, effTrue, 'NH', ni, nt)
    
    return Tdata, effTrue 
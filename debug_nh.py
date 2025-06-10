"""
Debug script for NH estimator
"""
import numpy as np
from dgp_nh import dgp_nh
from estimator_nh import nh_estimator
import pandas as pd

# Generate test data
print("="*80, flush=True)
print("Generating NH test data for a single run...", flush=True)
Tdata, eff_true = dgp_nh(
    ni=50, nt=5, 
    bf_true=[0.5, 0.5, 1.0], 
    bu_true=[0.5, -0.5], 
    bv_true=[-0.5, 0.5], 
    validate=True
)
print("="*80, flush=True)

print("\nTesting the refactored nh_estimator...", flush=True)
try:
    # Use keyword arguments for clarity and test new features
    result = nh_estimator(
        data=Tdata,
        ny_var='y',
        xf_vars=['xf1', 'xf2'],
        xu_vars=['xu'],
        xv_vars=['xv'],
        s=1,
        se_method='both',  # Test calculation of both SE types
        num_starts=2,
        save_path='debug_nh_output.xlsx', # Test saving functionality
        verbose=True
    )
    
    if result.get('converged', False):
        print("\n--- DEBUG SCRIPT: Final Checks ---", flush=True)
        print(f"Log-likelihood: {result['loglik']:.4f}", flush=True)
        
        # Check efficiency correlation
        eff_hat = result.get('eff_hat', np.array([np.nan]))
        if np.all(np.isfinite(eff_hat)) and np.all(np.isfinite(eff_true)):
            corr = np.corrcoef(eff_hat, eff_true)[0, 1]
            print(f"Correlation between True and Estimated Efficiency: {corr:.4f}", flush=True)
            if corr < 0.9:
                print("WARNING: Efficiency correlation is lower than expected.", flush=True)
        else:
            print("Could not compute efficiency correlation due to NaNs.", flush=True)
            
        print("\nSE Methods Check:", flush=True)
        print(f"OPG SEs available: {result.get('se_opg') is not None}", flush=True)
        print(f"Hessian SEs available: {result.get('se_hessian') is not None}", flush=True)
        print("--- End of Debug Script ---", flush=True)

    else:
        print("\n--- DEBUG SCRIPT: Estimation Failed ---", flush=True)
        
except Exception as e:
    print(f"\n--- DEBUG SCRIPT: An error occurred during testing ---", flush=True)
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc() 
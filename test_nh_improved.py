"""
Test script for improved NH estimator with nice output formatting
"""

from dgp_nh import dgp_nh
from estimator_nh import nh_estimator
import numpy as np

# Set seed for reproducibility
np.random.seed(123)

print("Testing NH Estimator with Improved Formatting")
print("=" * 60)

# Generate test data with the same parameters as simulation
print("\n1. Generating test data...")
Tdata, effTrue = dgp_nh(500, 8, np.array([0.5, 0.5, 1.0]), np.array([2.0, -0.5]), np.array([1.0, -0.5]))

print(f"   Data shape: {Tdata.shape}")
print(f"   True efficiency: mean={effTrue.mean():.3f}, std={effTrue.std():.3f}")

# Test production frontier (default)
print("\n2. Estimating PRODUCTION frontier...")
result_prod = nh_estimator(Tdata, ny=1, nf=2, nu=1, nv=1, verbose=True)

# Test cost frontier
print("\n3. Estimating COST frontier...")
result_cost = nh_estimator(Tdata, ny=1, nf=2, nu=1, nv=1, cost=True, verbose=True)

# Compare efficiency correlations
if result_prod['converged'] and result_cost['converged']:
    prod_corr = np.corrcoef(effTrue, result_prod['effhat'])[0, 1]
    cost_corr = np.corrcoef(effTrue, result_cost['effhat'])[0, 1]
    
    print(f"\n4. COMPARISON:")
    print(f"   Production frontier efficiency correlation: {prod_corr:.3f}")
    print(f"   Cost frontier efficiency correlation: {cost_corr:.3f}")
    print(f"   Better model: {'Production' if prod_corr > cost_corr else 'Cost'}") 
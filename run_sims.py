"""
Streamlined Python Simulation Framework for Stochastic Frontier Analysis Models

This is the main simulation driver that coordinates DGPs and estimators.
Easy-to-control settings are at the top of the file.

Authors: Based on MATLAB code by Kutlu and Sickles (2022)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import warnings
import os
from datetime import datetime
from joblib import Parallel, delayed
import argparse

# Import DGP functions
from dgp_nh import dgp_nh
from dgp_tre import dgp_tre  
from dgp_twre import dgp_twre, dgp_mtwre

# Import estimator functions
from estimator_nh import nh_estimator
# from estimator_tre import tre_estimator

# =============================================================================
# --- MAIN SIMULATION CONFIGURATION ---
# All simulation parameters can be easily modified here.
# Command-line arguments will override these settings.
# =============================================================================
SIMULATION_CONFIG = {
    # --- Model Selection ---
    'model': 'NH',  # DGP model to simulate. E.g., 'NH', 'TRE', 'TWRE'
    
    # --- Monte Carlo Settings ---
    'replications': 400,
    'parallel': True, # Use True for parallel processing, False for serial
    
    # --- Model Configuration ---
    'frontier_type': 'cost',  # 'prod' or 'cost'
    'se_method': 'opg',       # 'opg' or 'hessian'

    # --- Data Generation Parameters ---
    'ni': 200, # Number of firms
    'nt': 8,   # Number of time periods
    
    # True Parameters for the Data Generating Process
    # These are for log(variance), not log(stdev)
    'bf_true': np.array([0.5, 0.5, 1.0]),
    'bu_true': np.array([2.0, -0.5]),
    'bv_true': np.array([1.0, -0.5]),

    # Parameters for other models (not used by NH)
    'het': 0.3,
    'rho': 0.8,
}

# =============================================================================
# SCRIPT CORE LOGIC
# (Should not require modification for typical simulation runs)
# =============================================================================

# Maps a DGP model name to the name of the estimator function to use
DGP_ESTIMATOR_MAP = {
    'NH': 'NH',
    # 'TRE': 'TRE',  # Example for when other models are added
    # 'TWRE': 'TWRE',
}

# =============================================================================
# SIMULATION CORE FUNCTIONS
# =============================================================================

def run_single_replication(dgp_name, estimator_name, sim_params, rep_idx):
    """Run a single Monte Carlo replication."""
    try:
        # Generate data using appropriate DGP
        Tdata, effTrue = dgp_nh(
            sim_params['ni'], sim_params['nt'], 
            sim_params['bf_true'], sim_params['bu_true'], sim_params['bv_true'],
            s=sim_params['s']
        )
        
        # Estimate using appropriate estimator
        result = nh_estimator(
            data=Tdata,
            s=sim_params['s'],
            se_method=sim_params['se_method'],
            verbose=False
        )
        
        # Add true efficiency and other info
        result['effTrue'] = effTrue
        result['dgp_name'] = dgp_name
        result['estimator_name'] = estimator_name
        result['rep_num'] = rep_idx
        # Add a generic 'se' key for downstream compatibility in stats
        se_key = 'se_hessian' if sim_params['se_method'] == 'hessian' else 'se_opg'
        result['se'] = result.get(se_key)
        
        return result
        
    except Exception as e:
        warnings.warn(f"Replication {rep_idx} failed: {e}")
        return {
            'converged': False,
            'dgp_name': dgp_name,
            'estimator_name': estimator_name,
            'rep_num': rep_idx,
            'error': str(e)
        }

def compute_simulation_statistics(results, true_params):
    """Compute comprehensive simulation statistics including empirical standard errors."""
    successful = [r for r in results if r['converged']]
    n_success = len(successful)
    n_total = len(results)
    
    stats = {
        'n_total': n_total,
        'n_successful': n_success,
        'success_rate': n_success / n_total if n_total > 0 else 0
    }
    
    if n_success == 0:
        return stats
    
    # Extract parameter estimates and standard errors
    bhat_list = [r['bhat'] for r in successful]
    
    # Choose the correct SE based on the simulation's configuration
    se_key = 'se_hessian' if true_params['se_method'] == 'hessian' else 'se_opg'
    se_list = [r.get(se_key) for r in successful]

    # Handle cases where the chosen SE method failed for some replications
    valid_indices = [i for i, se in enumerate(se_list) if se is not None and np.all(np.isfinite(se))]
    if len(valid_indices) < n_success:
        warnings.warn(f"Found {n_success - len(valid_indices)} reps with missing/invalid SEs. They will be excluded from inference stats.", UserWarning)
        successful = [successful[i] for i in valid_indices]
        bhat_list = [bhat_list[i] for i in valid_indices]
        se_list = [se_list[i] for i in valid_indices]
        n_success = len(successful)
        if n_success == 0: return stats

    # Convert to arrays
    bhat_array = np.array(bhat_list)
    se_array = np.array(se_list)
    
    # True parameter vector (handle different models)
    true_vec = np.concatenate([true_params['bf_true'], true_params['bu_true'], true_params['bv_true']])
    
    # For TRE model, add the random effect variance parameter (log σ_α²)
    if len(successful) > 0 and 'dgp_name' in successful[0] and successful[0]['dgp_name'] == 'TRE':
        # For TRE, add log(σ_α²) = log(het²) where het is the heterogeneity parameter
        log_sigma_alpha_true = np.log(true_params['het']**2)
        true_vec = np.concatenate([true_vec, [log_sigma_alpha_true]])
    
    # Compute statistics
    stats['mean_estimates'] = np.mean(bhat_array, axis=0)
    stats['bias'] = stats['mean_estimates'] - true_vec
    stats['rmse'] = np.sqrt(np.mean((bhat_array - true_vec)**2, axis=0))
    stats['std_estimates'] = np.std(bhat_array, axis=0)  # Empirical standard errors
    stats['mean_se'] = np.mean(se_array, axis=0)        # Mean asymptotic standard errors
    
    # Coverage probabilities for 95% confidence intervals (5% significance level)
    z_critical = 1.96  # 5% two-sided test
    coverage = []
    for i in range(len(true_vec)):
        # Check if true parameter is within 95% CI for each replication
        lower_bounds = bhat_array[:, i] - z_critical * se_array[:, i]
        upper_bounds = bhat_array[:, i] + z_critical * se_array[:, i]
        in_ci = (lower_bounds <= true_vec[i]) & (true_vec[i] <= upper_bounds)
        coverage.append(np.mean(in_ci))
    stats['coverage_95'] = np.array(coverage)
    
    # Size of tests (empirical rejection rate when null is true)
    # For testing H0: parameter = true_value vs H1: parameter != true_value
    t_stats = (bhat_array - true_vec) / se_array
    rejection_rate = np.mean(np.abs(t_stats) > z_critical, axis=0)
    stats['rejection_rate_5pct'] = rejection_rate
    
    # Efficiency statistics
    if 'effTrue' in successful[0] and 'effhat' in successful[0]:
        effTrue_list = [r['effTrue'] for r in successful]
        effhat_list = [r['effhat'] for r in successful]
        
        # Compute correlations
        correlations = []
        for i in range(n_success):
            if not np.any(np.isnan(effhat_list[i])):
                corr = np.corrcoef(effTrue_list[i], effhat_list[i])[0, 1]
                if np.isfinite(corr):
                    correlations.append(corr)
        
        stats['mean_eff_correlation'] = np.mean(correlations) if correlations else np.nan
        stats['std_eff_correlation'] = np.std(correlations) if correlations else np.nan
    
    return stats

def run_dgp_simulation(dgp_name, params):
    """Run simulation for a specific DGP."""
    print(f"\n{'='*60}", flush=True)
    print(f"Running {dgp_name} DGP Simulation with {params['frontier_type_str']} frontier and {params['se_method']} SEs", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Define which estimator to use for the given DGP model
    estimator_map = {'NH': 'NH', 'TRE': 'TRE', 'TWRE': 'TWRE', 'MTWRE': 'MTWRE'}
    estimator_name = estimator_map.get(dgp_name)
    if not estimator_name:
        raise ValueError(f"No valid estimator found for DGP: {dgp_name}")

    print(f"\nRunning {estimator_name} estimator on {dgp_name} DGP...", flush=True)
    
    all_results = []
    
    # Choose execution path for a visible progress bar in both cases
    if params['n_jobs'] != 1:
        # Parallel execution
        all_results = Parallel(n_jobs=params['n_jobs'])(
            delayed(run_single_replication)(dgp_name, estimator_name, params, i)
            for i in tqdm(range(params['replications']))
        )
    else:
        # Serial execution
        for i in tqdm(range(params['replications'])):
            result = run_single_replication(dgp_name, estimator_name, params, i)
            all_results.append(result)

    # --- Post-simulation Analysis ---
    # Compare estimates to the original true parameters directly.
    # The estimator should correctly handle the sign of `s` internally.
    stats = compute_simulation_statistics(all_results, params)
    
    # Store results
    dgp_results = {
        estimator_name: {
            'results': all_results,
            'statistics': stats
        }
    }
    
    # Print nicely formatted summary
    print_simulation_summary(estimator_name, dgp_name, stats, params)
    
    return dgp_results

def main():
    """Main function to run the simulation based on config and command-line args."""
    # --- Setup Configuration ---
    # Start with the default config and override with any command-line arguments
    parser = argparse.ArgumentParser(description="Stochastic Frontier Model Simulation Framework")
    parser.add_argument('--model', type=str, default=SIMULATION_CONFIG['model'], help=f"Model to simulate (default: {SIMULATION_CONFIG['model']})")
    parser.add_argument('--replications', type=int, default=SIMULATION_CONFIG['replications'], help=f"Number of replications (default: {SIMULATION_CONFIG['replications']})")
    parser.add_argument('--frontier_type', type=str, default=SIMULATION_CONFIG['frontier_type'], choices=['prod', 'cost'], help="Type of frontier ('prod' or 'cost')")
    parser.add_argument('--se_method', type=str, default=SIMULATION_CONFIG['se_method'], choices=['opg', 'hessian'], help="SE calculation method")
    parser.add_argument('--parallel', action='store_true', default=SIMULATION_CONFIG['parallel'], help='Enable parallel processing.')
    args = parser.parse_args()

    # Create the final config by layering: 1. Defaults, 2. User-set, 3. CLI args
    config = SIMULATION_CONFIG.copy()
    # Filter out None values from CLI args to not override settings with defaults
    cli_args = {k: v for k, v in vars(args).items() if v is not None and (k != 'parallel' or v is True)}
    config.update(cli_args)

    # Add derived parameters for internal use
    config['s'] = -1 if config['frontier_type'] == 'cost' else 1
    config['frontier_type_str'] = config['frontier_type'] # for display
    config['n_jobs'] = -1 if config['parallel'] else 1

    # --- Run Simulation ---
    print("="*80, flush=True)
    print("STOCHASTIC FRONTIER ANALYSIS SIMULATION FRAMEWORK")
    print("="*80, flush=True)
    print("\nSimulation Configuration:")
    print(f"  Model (DGP): {config['model']}")
    print(f"  Replications: {config['replications']}")
    print(f"  Frontier Type: {config['frontier_type']} (s={config['s']})")
    print(f"  SE Method: {config['se_method']}")
    print(f"  Parallel: {'Yes' if config['parallel'] else 'No'}")
    print(f"  Sample size: {config['ni']} firms × {config['nt']} periods = {config['ni']*config['nt']} obs")
    print("\nTrue Parameters:")
    print(f"  Frontier (β): {config['bf_true']}")
    print(f"  Inefficiency Var (δ): {config['bu_true']}")
    print(f"  Noise Var (γ): {config['bv_true']}")

    # Dispatch to the correct DGP simulation runner
    all_dgp_results = {
        config['model']: run_dgp_simulation(config['model'], config)
    }

    # Save results to a timestamped Excel file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config['model']}_simulation_results_{timestamp}.xlsx"
    save_simulation_results(all_dgp_results, filename_prefix=filename)
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETED")
    print("="*80, flush=True)

def print_simulation_summary(estimator_name, dgp_name, stats, true_params):
    """Print nicely formatted simulation summary statistics."""
    
    success_rate = stats.get('success_rate', 0)
    n_successful = stats.get('n_successful', 0)
    n_total = stats.get('n_total', 0)
    
    eff_corr_mean = stats.get('mean_eff_correlation', np.nan)
    eff_corr_std = stats.get('std_eff_correlation', np.nan)

    print(f"\n{'='*70}", flush=True)
    print(f"{estimator_name} ESTIMATOR ON {dgp_name} DGP - SIMULATION SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Convergence Rate: {success_rate:.1%} ({n_successful}/{n_total} successful)", flush=True)

    if not np.isnan(eff_corr_mean):
        print(f"Efficiency Correlation: {eff_corr_mean:.3f} ± {eff_corr_std:.3f}", flush=True)
        quality = "Excellent" if eff_corr_mean > 0.95 else "Good" if eff_corr_mean > 0.8 else "Fair"
        icon = "✅" if quality in ["Excellent", "Good"] else "⚠️"
        print(f"{icon} {quality} efficiency prediction", flush=True)
        
    if 'mean_estimates' not in stats:
        print("\nNo successful replications to analyze.", flush=True)
        print("="*70, flush=True)
        return

    # Parameter names
    param_names = [
        'β₁ (xf1)', 'β₂ (xf2)', 'β₀ (const)',
        'δ₁ (xu)', 'δ₂ (const)',
        'γ₁ (xv)', 'γ₂ (const)'
    ]
    # For TRE, add the extra parameter
    if estimator_name == 'TRE':
        param_names.append('log(σ_α²)')

    true_vec = np.concatenate([true_params['bf_true'], true_params['bu_true'], true_params['bv_true']])
    if estimator_name == 'TRE':
        log_sigma_alpha_true = np.log(true_params['het']**2)
        true_vec = np.concatenate([true_vec, [log_sigma_alpha_true]])
    
    # --- Parameter Estimation Results Table ---
    print("\n" + "-"*70, flush=True)
    print("PARAMETER ESTIMATION RESULTS", flush=True)
    print("-"*70, flush=True)
    print(f"{'Parameter':<15} {'True Value':>12} {'Mean Est.':>12} {'Std Dev.':>12} {'Bias':>10} {'RMSE':>10} {'Quality':>10}", flush=True)
    print("-"*70, flush=True)

    for i, name in enumerate(param_names):
        true = true_vec[i]
        mean_est = stats['mean_estimates'][i]
        std_dev = stats['std_estimates'][i] # This is the empirical SE
        bias = stats['bias'][i]
        rmse = stats['rmse'][i]
        
        # Quality assessment based on bias relative to true value
        # (avoid division by zero for true values of 0)
        relative_bias = np.abs(bias / true) if np.abs(true) > 1e-6 else np.abs(bias)
        quality = "Excellent" if relative_bias < 0.05 else "Good" if relative_bias < 0.15 else "Fair" if relative_bias < 0.3 else "Poor"
        icon = "✅" if quality in ["Excellent", "Good"] else "⚠️" if quality == "Fair" else "❌"
        
        print(f"{name:<15} {true:>12.3f} {mean_est:>12.3f} {std_dev:>12.4f} {bias:>10.3f} {rmse:>10.4f}   {icon} {quality}", flush=True)
        
    # --- Inference Results Table ---
    print("\n" + "-"*70, flush=True)
    print("EMPIRICAL STANDARD ERRORS & INFERENCE (5% Significance Level)", flush=True)
    print("-"*70, flush=True)
    print(f"{'Parameter':<15} {'Empirical SE':>14} {'Asymptotic SE':>15} {'Coverage 95%':>15} {'Rejection Rate':>18}", flush=True)
    print("-"*70, flush=True)
    
    for i, name in enumerate(param_names):
        emp_se = stats['std_estimates'][i]
        asy_se = stats['mean_se'][i]
        coverage = stats['coverage_95'][i]
        rejection = stats['rejection_rate_5pct'][i]
        
        # Coverage quality
        cov_quality = "✅" if 0.925 <= coverage <= 0.975 else "⚠️" if 0.90 <= coverage < 0.925 or 0.975 < coverage <= 0.99 else "❌"
        # Rejection rate (size) quality
        rej_quality = "✅" if 0.025 <= rejection <= 0.075 else "⚠️" if 0.01 <= rejection < 0.025 or 0.075 < rejection <= 0.10 else "❌"
        
        print(f"{name:<15} {emp_se:>14.4f} {asy_se:>15.4f} {coverage:>14.3f} {cov_quality} {rejection:>15.3f} {rej_quality}", flush=True)

    print("\nInference Quality: ✅ Good (coverage ≈ 95%, size ≈ 5%), ⚠️ Fair, ❌ Poor", flush=True)
    print("="*70, flush=True)

def save_simulation_results(all_results, filename_prefix="simulation_results"):
    """Save simulation results to Excel format with multiple sheets."""
    import pandas as pd
    
    # Use the filename passed from main(), which already has a timestamp
    filename = filename_prefix 

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # --- Create a single summary sheet ---
            summary_data = []
            param_names = ['β₁', 'β₂', 'β₀', 'δ₁', 'δ₂', 'γ₁', 'γ₂']

            for dgp_name, dgp_results in all_results.items():
                for estimator_name, est_results in dgp_results.items():
                    stats = est_results['statistics']
                    row = {
                        'DGP': dgp_name,
                        'Estimator': estimator_name,
                        'Convergence_Rate': stats.get('success_rate', 0),
                        'N_Successful': stats.get('n_successful', 0),
                        'Efficiency_Correlation': stats.get('mean_eff_correlation', np.nan)
                    }
                    # Add stats for each parameter
                    for i, name in enumerate(param_names):
                        if i < len(stats.get('mean_estimates', [])):
                            row[f'{name}_Bias'] = stats['bias'][i]
                            row[f'{name}_RMSE'] = stats['rmse'][i]
                            row[f'{name}_EmpSE'] = stats['std_estimates'][i]
                            row[f'{name}_AsymSE'] = stats['mean_se'][i]
                            row[f'{name}_Coverage95'] = stats['coverage_95'][i]
                    summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # --- Create detailed sheets for each run ---
            for dgp_name, dgp_results in all_results.items():
                for estimator_name, est_results in dgp_results.items():
                    successful_results = [r for r in est_results.get('results', []) if r.get('converged')]
                    if not successful_results: continue

                    # Prepare data for DataFrame
                    detailed_data = []
                    for res in successful_results:
                        # Correlation for this single run
                        corr = np.nan
                        if 'effTrue' in res and 'effhat' in res and not np.any(np.isnan(res['effhat'])):
                           corr_matrix = np.corrcoef(res['effTrue'], res['effhat'])
                           if corr_matrix.shape == (2, 2):
                               corr = corr_matrix[0, 1]
                        
                        # Base data for the row
                        row_data = {'LogLik': res.get('loglik'), 'Eff_Corr': corr}
                        
                        # Add bhat and se values
                        for i, name in enumerate(param_names):
                            row_data[f'{name}_bhat'] = res['bhat'][i] if i < len(res.get('bhat', [])) else np.nan
                            row_data[f'{name}_se'] = res['se'][i] if res.get('se') is not None and i < len(res['se']) else np.nan
                        
                        detailed_data.append(row_data)

                    detailed_df = pd.DataFrame(detailed_data)
                    sheet_name = f"{dgp_name}_{estimator_name}"[:31]
                    if not detailed_df.empty:
                        detailed_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\n📊 Simulation results saved to: {filename}", flush=True)
        return filename
        
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}", flush=True)
        # Fallback to pickle
        import pickle
        pkl_filename = filename_prefix.replace('.xlsx', '.pkl')
        with open(pkl_filename, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"📁 Saved to pickle format instead: {pkl_filename}", flush=True)
        return pkl_filename

if __name__ == '__main__':
    main() 
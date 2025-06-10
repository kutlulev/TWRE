"""
Enhanced Result Aggregation and Output Module for SFA Simulations

This module aggregates, summarizes, and exports simulation results with:
- Comprehensive bias, RMSE, and correlation statistics
- CSV and Excel export capabilities
- Detailed efficiency analysis
- Statistical significance testing

Authors: Based on MATLAB code by Kutlu and Sickles (2022)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import pickle
import os
from datetime import datetime
import warnings

def _safe_statistic(data, func, default=np.nan):
    """Safely compute statistics with error handling."""
    try:
        if len(data) == 0:
            return default
        return func(data)
    except:
        return default

def _fill_missing_params(results, estimator, param_type, p_len, sim_n):
    """Helper to extract and fill parameter arrays."""
    mat = np.full((p_len, sim_n), np.nan)
    for s, res in enumerate(results):
        if estimator in res and res[estimator]:
            param_mat = res[estimator].get(param_type)
            if param_mat is not None and param_mat.shape[0] >= p_len:
                mat[:, s] = param_mat[:p_len, 0]  # First column contains estimates
    return mat

def _calculate_metrics(param_mat, true_params):
    """Helper to compute mean, bias, RMSE, and other statistics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Basic statistics
        mean_est = np.nanmean(param_mat, axis=1)
        bias = mean_est - true_params
        rmse_est = np.sqrt(np.nanmean((param_mat - true_params[:, np.newaxis])**2, axis=1))
        std_est = np.nanstd(param_mat, axis=1)
        
        # Additional metrics
        median_est = np.nanmedian(param_mat, axis=1)
        q25_est = np.nanpercentile(param_mat, 25, axis=1)
        q75_est = np.nanpercentile(param_mat, 75, axis=1)
        
        return {
            'mean': mean_est,
            'bias': bias,
            'rmse': rmse_est,
            'std': std_est,
            'median': median_est,
            'q25': q25_est,
            'q75': q75_est
        }

def _efficiency_analysis(results, estimators_list):
    """Comprehensive efficiency analysis across estimators."""
    eff_results = {}
    
    for est in estimators_list:
        diffs = []
        corrs = []
        mean_effs = []
        med_effs = []
        std_effs = []
        
        for res in results:
            if res.get('true_eff') is not None and est in res and res[est]:
                true_eff = res['true_eff']
                est_eff = res[est]['eff']
                
                if true_eff is not None and est_eff is not None:
                    n = min(len(true_eff), len(est_eff))
                    if n > 0:
                        true_subset = true_eff[:n]
                        est_subset = est_eff[:n]
                        
                        # Remove any NaN or infinite values
                        valid_mask = np.isfinite(true_subset) & np.isfinite(est_subset)
                        if np.sum(valid_mask) > 5:  # Need minimum observations
                            true_valid = true_subset[valid_mask]
                            est_valid = est_subset[valid_mask]
                            
                            diffs.extend(est_valid - true_valid)
                            
                            # Correlation
                            try:
                                corr, _ = spearmanr(true_valid, est_valid)
                                if np.isfinite(corr):
                                    corrs.append(corr)
                            except:
                                pass
                            
                            # Summary statistics
                            mean_effs.append(np.mean(est_valid))
                            med_effs.append(np.median(est_valid))
                            std_effs.append(np.std(est_valid))
        
        # Aggregate results
        eff_results[est] = {
            'mean_eff': _safe_statistic(mean_effs, np.mean),
            'median_eff': _safe_statistic(med_effs, np.mean),
            'std_eff': _safe_statistic(std_effs, np.mean),
            'bias_eff': _safe_statistic(diffs, np.mean),
            'rmse_eff': _safe_statistic(diffs, lambda x: np.sqrt(np.mean(np.array(x)**2))),
            'std_bias_eff': _safe_statistic(diffs, np.std),
            'spearman_corr': _safe_statistic(corrs, np.mean),
            'spearman_std': _safe_statistic(corrs, np.std),
            'n_successful': len(mean_effs)
        }
    
    return eff_results

def _create_parameter_summary(params_dict, true_params_dict, param_names):
    """Create comprehensive parameter summary tables."""
    summary_tables = {}
    
    for param_group, true_vals in true_params_dict.items():
        if param_group not in params_dict:
            continue
            
        estimators = list(params_dict[param_group].keys())
        n_params = len(true_vals)
        
        # Initialize summary dataframe
        columns = ['True_Value', 'Mean', 'Bias', 'RMSE', 'Std_Dev', 'Median', 'Q25', 'Q75']
        index = [param_names[param_group][i] for i in range(n_params)]
        
        summary_data = {}
        
        for est in estimators:
            param_mat = params_dict[param_group][est]
            metrics = _calculate_metrics(param_mat, true_vals)
            
            for i in range(n_params):
                param_name = index[i]
                
                if param_name not in summary_data:
                    summary_data[param_name] = {
                        'True_Value': true_vals[i],
                        f'{est}_Mean': metrics['mean'][i],
                        f'{est}_Bias': metrics['bias'][i],
                        f'{est}_RMSE': metrics['rmse'][i],
                        f'{est}_Std': metrics['std'][i],
                        f'{est}_Median': metrics['median'][i],
                        f'{est}_Q25': metrics['q25'][i],
                        f'{est}_Q75': metrics['q75'][i]
                    }
                else:
                    summary_data[param_name].update({
                        f'{est}_Mean': metrics['mean'][i],
                        f'{est}_Bias': metrics['bias'][i],
                        f'{est}_RMSE': metrics['rmse'][i],
                        f'{est}_Std': metrics['std'][i],
                        f'{est}_Median': metrics['median'][i],
                        f'{est}_Q25': metrics['q25'][i],
                        f'{est}_Q75': metrics['q75'][i]
                    })
        
        summary_tables[param_group] = pd.DataFrame.from_dict(summary_data, orient='index')
    
    return summary_tables

def _export_results(results_dict, output_dir, dgp_type):
    """Export results to CSV and Excel formats."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{dgp_type}_simulation_results_{timestamp}"
    
    # Excel export with multiple sheets
    excel_path = os.path.join(output_dir, f"{base_filename}.xlsx")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # Parameter summary sheets
            if 'parameter_summaries' in results_dict:
                for param_group, df in results_dict['parameter_summaries'].items():
                    sheet_name = f"{param_group}_params"[:31]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
            
            # Efficiency summary
            if 'efficiency_summary' in results_dict:
                results_dict['efficiency_summary'].to_excel(writer, sheet_name="Efficiency", index=True)
            
            # Raw simulation info
            if 'simulation_info' in results_dict:
                info_df = pd.DataFrame([results_dict['simulation_info']])
                info_df.to_excel(writer, sheet_name="Simulation_Info", index=False)
        
        print(f"✓ Excel results exported to: {excel_path}")
        
    except Exception as e:
        print(f"⚠️  Excel export failed: {e}")
    
    # CSV exports
    try:
        csv_dir = os.path.join(output_dir, f"{base_filename}_csv")
        os.makedirs(csv_dir, exist_ok=True)
        
        # Export each table as separate CSV
        if 'parameter_summaries' in results_dict:
            for param_group, df in results_dict['parameter_summaries'].items():
                csv_path = os.path.join(csv_dir, f"{param_group}_parameters.csv")
                df.to_csv(csv_path, index=True)
        
        if 'efficiency_summary' in results_dict:
            eff_path = os.path.join(csv_dir, "efficiency_summary.csv")
            results_dict['efficiency_summary'].to_csv(eff_path, index=True)
        
        print(f"✓ CSV results exported to: {csv_dir}")
        
    except Exception as e:
        print(f"⚠️  CSV export failed: {e}")

def finalize_simulation_output(results, params):
    """
    Enhanced simulation results aggregation and export.
    """
    print("\n" + "="*60)
    print("FINALIZING SIMULATION RESULTS")
    print("="*60)
    
    sim_n = params['sim_n']
    dgp_type = params['dgp_type']
    estimators_list = params['estimators_list']
    
    # True parameter values
    true_params_dict = {
        'frontier': params['bf_true'],
        'inefficiency': params['bu_true'], 
        'noise': params['bv_true']
    }
    
    # Parameter names
    pf, pu, pv = len(params['bf_true']), len(params['bu_true']), len(params['bv_true'])
    param_names = {
        'frontier': [f'beta_{i+1}' for i in range(pf)],
        'inefficiency': [f'bu_{i+1}' for i in range(pu)],
        'noise': [f'bv_{i+1}' for i in range(pv)]
    }
    
    print(f"DGP Type: {dgp_type}")
    print(f"Estimators: {estimators_list}")
    print(f"Total replications: {sim_n}")
    
    # --- Count successful runs ---
    successful_runs = {est: 0 for est in estimators_list}
    total_runs = len(results)
    
    for res in results:
        for est in estimators_list:
            if est in res and res[est] and res[est].get('exitflag', -1) in [0, 1]:
                successful_runs[est] += 1

    print(f"\nSuccess Rates:")
    for est, count in successful_runs.items():
        success_rate = count / total_runs * 100
        print(f"  {est}: {count}/{total_runs} ({success_rate:.1f}%)")
    
    # --- Aggregate parameter estimates ---
    params_dict = {}
    
    for param_group, true_vals in true_params_dict.items():
        params_dict[param_group] = {}
        param_type = {'frontier': 'bf', 'inefficiency': 'bu', 'noise': 'bv'}[param_group]
        p_len = len(true_vals)
        
        for est in estimators_list:
            param_mat = _fill_missing_params(results, est, param_type, p_len, total_runs)
            params_dict[param_group][est] = param_mat

    # --- Create parameter summary tables ---
    parameter_summaries = _create_parameter_summary(params_dict, true_params_dict, param_names)
    
    # --- Efficiency analysis ---
    eff_analysis = _efficiency_analysis(results, estimators_list)
    
    # Create efficiency summary DataFrame
    eff_summary_data = {}
    for est, metrics in eff_analysis.items():
        eff_summary_data[est] = metrics
    
    eff_df = pd.DataFrame(eff_summary_data).T
    
    # --- Display results ---
    print(f"\n" + "="*60)
    print("PARAMETER ESTIMATION RESULTS")
    print("="*60)
    
    for param_group, summary_df in parameter_summaries.items():
        print(f"\n{param_group.upper()} PARAMETERS:")
        print("-" * 50)
        
        # Display with nice formatting
        display_columns = ['True_Value']
        for est in estimators_list:
            display_columns.extend([f'{est}_Mean', f'{est}_Bias', f'{est}_RMSE'])
        
        if all(col in summary_df.columns for col in display_columns):
            display_df = summary_df[display_columns]
            print(display_df.round(4).to_string())
        else:
            print(summary_df.round(4).to_string())
    
    print(f"\n" + "="*60)
    print("EFFICIENCY ANALYSIS")
    print("="*60)
    
    # Format efficiency results for display
    eff_display_cols = ['mean_eff', 'bias_eff', 'rmse_eff', 'spearman_corr', 'n_successful']
    if all(col in eff_df.columns for col in eff_display_cols):
        print(eff_df[eff_display_cols].round(4).to_string())
    else:
        print(eff_df.round(4).to_string())
    
    # --- Prepare results for export ---
    results_for_export = {
        'parameter_summaries': parameter_summaries,
        'efficiency_summary': eff_df,
        'simulation_info': {
            'dgp_type': dgp_type,
            'estimators': estimators_list,
            'total_replications': total_runs,
            'sample_size_n': params.get('ni', 'Unknown'),
            'sample_size_t': params.get('nt', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'success_rates': successful_runs
        },
        'raw_results': results  # Keep for detailed analysis
    }
    
    # --- Save results ---
    # Create output directory if it doesn't exist
    output_dir = "simulation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pickle file
    pickle_filename = f"{dgp_type}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    pickle_path = os.path.join(output_dir, pickle_filename)
    
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(results_for_export, f)
        print(f"\n✓ Pickle results saved to: {pickle_path}")
    except Exception as e:
        print(f"\n⚠️  Pickle save failed: {e}")
    
    # Export to CSV/Excel
    _export_results(results_for_export, output_dir, dgp_type)
    
    print(f"\n🎉 Results finalization completed for {dgp_type}")
    print(f"📁 All outputs saved in: {output_dir}")
    
    return results_for_export

if __name__ == '__main__':
    """
    Standalone testing mode - load existing results and re-process them.
    """
    try:
        # Look for existing pickle files
        pickle_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'results' in f]
        
        if pickle_files:
            latest_file = max(pickle_files, key=os.path.getctime)
            print(f"Loading results from: {latest_file}")
            
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)
            
            if 'raw_results' in data and 'simulation_info' in data:
                # Re-process with enhanced analysis
                params = {
                    'sim_n': data['simulation_info'].get('total_replications', 100),
                    'dgp_type': data['simulation_info'].get('dgp_type', 'Unknown'),
                    'estimators_list': data['simulation_info'].get('estimators', []),
                    'bf_true': np.array([0.5, 0.5, 1.0]),  # Default values
                    'bu_true': np.array([1.0, -0.5]),
                    'bv_true': np.array([1.0, -0.5]),
                    'ni': data['simulation_info'].get('sample_size_n', 100),
                    'nt': data['simulation_info'].get('sample_size_t', 8)
                }
                
                finalize_simulation_output(data['raw_results'], params)
            else:
                print("⚠️  Loaded data doesn't have the expected structure")
        else:
            print("⚠️  No result files found to process")
            
    except Exception as e:
        print(f"❌ Error in standalone processing: {e}")
        import traceback
        traceback.print_exc() 
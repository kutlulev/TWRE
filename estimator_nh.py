"""
No Heterogeneity (NH) Stochastic Frontier Model Estimator

This module provides robust estimation for standard SFA without random effects.
Based on the MATLAB code NH.m by Kutlu and Sickles (2022).
Enhanced with numerical stability techniques from ssfa_sem_hn.py.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import numdifftools as nd
import pandas as pd
import warnings
import time
from scipy.linalg import pinv, LinAlgError

# =============================================================================
# Main Estimator Function
# =============================================================================

def nh_estimator(
    data,
    ny_var='y',
    xf_vars=['xf1', 'xf2'],
    xu_vars=['xu'],
    xv_vars=['xv'],
    s=1,
    se_method='opg',
    num_starts=5,
    save_path=None,
    verbose=True,
    **kwargs
):
    """
    Production-ready Normal-Half Normal Stochastic Frontier Estimator.

    Args:
        data (pd.DataFrame): DataFrame containing all necessary variables.
        ny_var (str): Name of the dependent variable column.
        xf_vars (list): List of column names for frontier regressors.
        xu_vars (list): List of column names for inefficiency variance regressors.
        xv_vars (list): List of column names for noise variance regressors.
        s (int): Sign of the production model (1 for production, -1 for consumption).
        se_method (str): Method for SEs: 'opg', 'hessian', or 'both'. 'opg' is default.
        num_starts (int): Number of random starting points for optimization.
        save_path (str, optional): Path to save data with efficiencies ('...xlsx' or '...dta').
        verbose (bool): Whether to print estimation details and results.
        **kwargs: For rep_num in simulations or custom initial values.

    Returns:
        dict: A dictionary containing detailed estimation results.
    """
    if verbose:
        print("\n" + "="*80)
        print("Starting NH Stochastic Frontier Estimation...")
        print(f"Frontier type: Production (s={s}), SE Method: {se_method}, Num. Starts: {num_starts}")
        print(f"Data contains {data.shape[0]} observations.")
        print("="*80)

    # --- 1. Data Preparation ---
    try:
        y = data[ny_var].values
        # Automatically add intercept term to design matrices
        Xf = data[xf_vars].assign(const=1)[xf_vars + ['const']].values
        Xu = data[xu_vars].assign(const=1)[xu_vars + ['const']].values
        Xv = data[xv_vars].assign(const=1)[xv_vars + ['const']].values
        n_obs, kf, ku, kv = len(y), Xf.shape[1], Xu.shape[1], Xv.shape[1]
        k = kf + ku + kv
    except KeyError as e:
        raise ValueError(f"Missing required column in data: {e}")

    # --- 2. Define Log-Likelihood Functions ---
    def loglike_nh_obs(params, y, Xf, Xu, Xv, s):
        bf, bu, bv = params[:kf], params[kf:kf+ku], params[kf+ku:k]
        e = y - (Xf @ bf)

        # Parameterization is for log(variance), not log(stdev)
        log_sigu_sq = Xu @ bu
        log_sigv_sq = Xv @ bv
        log_sigu = 0.5 * log_sigu_sq
        log_sigv = 0.5 * log_sigv_sq

        # Use log-sum-exp trick to compute log(sigu**2 + sigv**2) safely
        a, b = log_sigu_sq, log_sigv_sq
        max_ab = np.maximum(a, b)
        log_sig_sq = max_ab + np.log(np.exp(a - max_ab) + np.exp(b - max_ab))

        # Term 1: -log(sig) = -0.5 * log_sig_sq
        term1 = -0.5 * log_sig_sq

        # Term 2: -0.5 * (e**2 / sig_sq)
        term2 = -0.5 * e**2 * np.exp(-log_sig_sq)

        # Term 3: log(cdf(z)) where z = s*e*lambda/sigma
        # This aligns exactly with the NH.m reference code.
        log_abs_z = np.log(np.abs(e) + 1e-100) + log_sigu - log_sigv - 0.5 * log_sig_sq
        z = s * np.sign(e) * np.exp(log_abs_z)
        term3 = norm.logcdf(z)

        # Combine all terms, including the log(2) from the half-normal density
        ll = np.log(2) + term1 + term2 + term3
        
        # Final safeguard against any remaining NaNs or Infs
        ll = np.nan_to_num(ll, nan=-1e12, posinf=-1e12, neginf=-1e12)
        return ll

    def loglike_nh(params, *args):
        return -np.sum(loglike_nh_obs(params, *args))

    # --- 3. Run Multi-Start Optimization ---
    best_loglik = -np.inf
    best_result = None
    
    start_vals_list = _get_robust_starting_values(num_starts, k, kf, y, Xf)

    for i, start_vals in enumerate(start_vals_list):
        if verbose: print(f"  Starting optimization {i+1}/{num_starts}... ", flush=True)
        try:
            # Use L-BFGS-B with bounds for stability
            bounds = [(None, None)] * kf + [(-10, 10)] * (ku + kv)

            res = minimize(
                loglike_nh,
                x0=start_vals,
                args=(y, Xf, Xu, Xv, s),
                method='L-BFGS-B',
                bounds=bounds,
                options={'disp': False, 'maxiter': 10000}
            )

            if res.success and -res.fun > best_loglik:
                best_loglik = -res.fun
                best_result = res
        except (ValueError, LinAlgError) as e:
            if verbose: print(f"    Optimization start {i+1} failed: {e}", flush=True)
            continue
            
    # --- 4. Process and Return Results ---
    if not best_result or not best_result.success:
        if verbose:
            warnings.warn("Optimization failed to converge after all attempts.", UserWarning)
        return {'converged': False, 'loglik':-best_result.fun if best_result else -np.inf, 'bhat': np.full(k, np.nan)}
    
    bhat = best_result.x
    loglik = -best_result.fun

    # --- 5. Post-estimation: Calculate VCV and Efficiency ---
    vcv_results = _calculate_vcov_robust(loglike_nh_obs, 
                                         loglike_nh, 
                                         bhat, (y, Xf, Xu, Xv, s), display=verbose)
    eff_results = _calculate_efficiency(bhat, y, Xf, Xu, Xv, s, kf, ku)

    # --- 6. Format and Return Results ---
    final_results = _format_results(
        bhat, loglik, vcv_results, eff_results, se_method,
        {'kf': kf, 'ku': ku, 'kv': kv}, verbose,
        {'xf_vars': xf_vars, 'xu_vars': xu_vars, 'xv_vars': xv_vars}
    )
    
    # --- 7. Save Data if Requested ---
    if save_path:
        _save_output_data(data, eff_results, save_path, verbose)
        
    return final_results

# =============================================================================
# Helper Functions
# =============================================================================

def _get_robust_starting_values(num_starts, k, kf, y, Xf):
    """Generate a list of starting values, with the first being from OLS."""
    start_vals_list = []
    
    # First start: OLS for betas, zeros for variance params
    try:
        b_ols = np.linalg.inv(Xf.T @ Xf) @ (Xf.T @ y)
        start_ols = np.zeros(k)
        start_ols[:kf] = b_ols
        start_vals_list.append(start_ols)
    except LinAlgError:
        # Fallback if OLS fails (e.g., perfect multicollinearity)
        start_vals_list.append(np.zeros(k))

    # Add random starts for the rest
    num_random_starts = num_starts - len(start_vals_list)
    if num_random_starts > 0:
        # Generate random values centered around 0
        random_starts = np.random.uniform(-0.1, 0.1, size=(num_random_starts, k))
        # For the first random start, try slightly perturbed OLS values
        if len(start_vals_list) > 0:
            random_starts[0, :] = start_vals_list[0] + np.random.uniform(-0.1, 0.1, k)
        
        start_vals_list.extend(list(random_starts))
        
    return start_vals_list

def _calculate_vcov_robust(loglik_obs_fn, loglik_fn, beta_hat, args, display=True):
    results = {}
    
    # --- OPG Calculation ---
    try:
        # Outer Product of Gradients (BHHH)
        jac = nd.Jacobian(loglik_obs_fn)(beta_hat, *args)
        opg = jac.T @ jac
        vcov_opg = np.linalg.inv(opg)
        se_opg = np.sqrt(np.diag(vcov_opg))
        results['opg'] = {'vcov': vcov_opg, 'se': se_opg}
    except (LinAlgError, ValueError) as e:
        if display: print(f"[VCov Warning] OPG calculation failed: {e}. Trying with regularization.")
        try:
            # Add a small identity matrix to improve conditioning
            opg_reg = opg + np.eye(opg.shape[0]) * 1e-6
            vcov_opg = np.linalg.inv(opg_reg)
            se_opg = np.sqrt(np.diag(vcov_opg))
            results['opg'] = {'vcov': vcov_opg, 'se': se_opg}
        except (LinAlgError, ValueError) as e2:
            if display: print(f"[VCov Warning] Regularized OPG also failed: {e2}")
            results['opg'] = None

    # --- Resilient Hessian Calculation ---
    try:
        hess_calc = nd.Hessian(lambda p: loglik_fn(p, *args), step=1e-6, method='central')
        hessian = hess_calc(beta_hat)
        hessian = (hessian + hessian.T) / 2
        try:
            vcov_hess = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            if display: print("[VCov Warning] Hessian not positive definite. Regularizing.")
            min_eig = np.min(np.linalg.eigvalsh(hessian))
            hessian += (-min_eig + 1e-8) * np.eye(len(beta_hat))
            vcov_hess = np.linalg.inv(hessian)
        results['hessian'] = {'se': np.sqrt(np.abs(np.diag(vcov_hess))), 'vcov': vcov_hess}
        if display: print("[VCov Info] Hessian standard errors calculated successfully.")
    except Exception as e:
        if display: print(f"[VCov Warning] Hessian calculation failed: {e}")
        results['hessian'] = None
        
    return results

def _calculate_efficiency(bhat, y, Xf, Xu, Xv, s, kf, ku):
    bf, bu, bv = bhat[:kf], bhat[kf:kf+ku], bhat[kf+ku:]
    
    e = y - (Xf @ bf)
    
    # Re-use the stable log-domain calculations with correct parameterization
    log_sigu_sq = Xu @ bu
    log_sigv_sq = Xv @ bv
    log_sigu = 0.5 * log_sigu_sq
    log_sigv = 0.5 * log_sigv_sq
    
    sigu_sq = np.exp(log_sigu_sq)
    sigv_sq = np.exp(log_sigv_sq)
    sig_sq = sigu_sq + sigv_sq

    # mu_star = -s*e*lambda^2 / (1+lambda^2) -> simplified form
    # This aligns exactly with the NH.m reference code.
    mu_star = -s * e * sigu_sq / (sig_sq + 1e-9)
    sig_star_sq = sigu_sq * sigv_sq / (sig_sq + 1e-9)
    sig_star = np.sqrt(sig_star_sq)

    z = mu_star / (sig_star + 1e-9)
    
    # Numerically stable calculation of u_hat = E(u|e) using the inverse Mills ratio
    # The ratio norm.pdf(z)/norm.cdf(z) is unstable for large negative z.
    # We compute it in the log domain to avoid division by zero.
    log_ratio = norm.logpdf(z) - norm.logcdf(z)
    ratio = np.exp(log_ratio)

    # The formula is u_hat = mu* + sig* * ratio
    u_hat = mu_star + sig_star * ratio
    eff_hat = np.exp(-u_hat)
    
    return {'u_hat': u_hat, 'eff_hat': eff_hat}

def _format_results(bhat, loglik, vcv_results, eff_results, se_method, dims, verbose, var_names):
    p_dims, v_names = dims, var_names
    
    results = {
        'converged': True, 'loglik': loglik, 'bhat': bhat,
        'u_hat': eff_results['u_hat'], 'eff_hat': eff_results['eff_hat'],
        'se_opg': (vcv_results.get('opg') or {}).get('se'),
        'se_hessian': (vcv_results.get('hessian') or {}).get('se'),
        'vcov_opg': (vcv_results.get('opg') or {}).get('vcov'),
        'vcov_hessian': (vcv_results.get('hessian') or {}).get('vcov'),
    }

    if verbose:
        param_names = {
            'Frontier': v_names['xf_vars'] + ['const_f'],
            'Inefficiency Var.': v_names['xu_vars'] + ['const_u'],
            'Noise Var.': v_names['xv_vars'] + ['const_v']
        }
        
        print_nh_results(
            bhat, loglik, 1, eff_results['eff_hat'],
            vcv_results, se_method, param_names, p_dims
        )
    return results

def _save_output_data(data, eff_results, save_path, display):
    output_df = data.copy()
    output_df['u_hat'] = eff_results['u_hat']
    output_df['eff_hat'] = eff_results['eff_hat']
    try:
        if save_path.lower().endswith('.dta'):
            output_df.to_stata(save_path, write_index=False)
        else:
            if not save_path.lower().endswith('.xlsx'): save_path += '.xlsx'
            output_df.to_excel(save_path, index=False)
        if display:
            print(f"\n--- Data with efficiency estimates saved to: {save_path} ---")
    except Exception as e:
        warnings.warn(f"Could not save output file: {e}", UserWarning)

def print_nh_results(bhat, loglik, s, effhat, vcv_results, se_method, param_names, p_dims):
    print("\n" + "="*80)
    print(f"NH STOCHASTIC FRONTIER ESTIMATION RESULTS (s={s})")
    print("="*80)
    print(f"Log-likelihood: {loglik:.4f}, Mean Efficiency: {np.mean(effhat):.4f}")
    print("-"*80)
    
    se_opg = vcv_results.get('opg', {}).get('se', np.full_like(bhat, np.nan))
    se_hess = vcv_results.get('hessian', {}).get('se', np.full_like(bhat, np.nan))

    header = f"{'Variable':<15} {'Coefficient':>15}"
    # Choose primary SE for p-value calculation based on availability
    primary_se_source = 'opg' if se_method != 'hessian' and np.all(np.isfinite(se_opg)) else 'hessian'
    
    if se_method in ['opg', 'both']: header += f"{'OPG SE':>15}"
    if se_method in ['hessian', 'both']: header += f"{'Hessian SE':>15}"
    header += f"{'P-value ({})':>16}".format(primary_se_source)
    print(header)
    print("-"*len(header))

    
    param_idx = 0
    for title, names in param_names.items():
        print(f"\n{title} Parameters:")
        for name in names:
            line = f"{name:<15} {bhat[param_idx]:>15.6f}"
            
            # Use specified SE for p-value
            se_for_pval = se_opg if primary_se_source == 'opg' else se_hess
            pval = 2 * (1 - norm.cdf(np.abs(bhat[param_idx] / se_for_pval[param_idx]))) if np.isfinite(se_for_pval[param_idx]) else np.nan
            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''

            if se_method in ['opg', 'both']:
                line += f" {se_opg[param_idx]:>15.6f}" if np.isfinite(se_opg[param_idx]) else f"{'nan':>15}"
            if se_method in ['hessian', 'both']:
                line += f" {se_hess[param_idx]:>15.6f}" if np.isfinite(se_hess[param_idx]) else f"{'nan':>15}"
            
            line += f" {pval:>12.4f}{stars}"
            print(line)
            param_idx += 1
    print("="*80)
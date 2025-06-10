"""
True Random Effects (TRE) Stochastic Frontier Model Estimator

This module provides robust estimation for SFA with random firm-specific frontier effects.
Based on the MATLAB code TRE.m by Kutlu and Sickles (2022).
Enhanced with numerical stability techniques from ssfa_sem_hn.py.
"""

import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.stats import norm, qmc
import warnings
import pandas as pd

# Import utility functions
class IterationLogger:
    """Silent iteration logger to avoid output during parallel processing."""
    def __init__(self, estimator_name, rep_num):
        self.estimator_name = estimator_name
        self.rep_num = rep_num
        self.iter = 0

    def __call__(self, xk):
        self.iter += 1
        # Silent - no output to avoid cluttering parallel processing

def _safe_pvalue_calculation(estimate, std_error, param_name="", display=True):
    """
    Calculates p-values with safeguards for numerical issues using a two-sided test.
    """
    if not np.isfinite(std_error) or std_error <= 1e-12:
        if display:
            print(f"[Warning] Standard error for {param_name} is invalid ({std_error:.2e}). P-value set to NaN.")
        return np.nan
    
    if not np.isfinite(estimate):
        if display:
            print(f"[Warning] Estimate for {param_name} is not finite ({estimate}). P-value set to NaN.")
        return np.nan
        
    t_stat = estimate / std_error
    pval = 2 * (1 - norm.cdf(np.abs(t_stat)))
    pval = np.clip(pval, 0.0, 1.0)
    return pval

def _calculate_vcov_safe(hess_inv_approx, neg_loglik_fun, final_params, display=True):
    """
    Calculates the variance-covariance matrix in a numerically safe way.
    Similar to the approach in ssfa_sem_hn.py with robust fallbacks.
    """
    num_params = len(final_params)
    
    def print_warning(msg):
        if display:
            print(f"[VCov Warning] {msg}")

    # 1. Try the inverse Hessian from the optimizer
    if hess_inv_approx is not None and isinstance(hess_inv_approx, np.ndarray):
        try:
            diag_vcov = np.diag(hess_inv_approx)
            if np.all(np.isfinite(diag_vcov)) and np.all(diag_vcov >= 0):
                vcov = hess_inv_approx
                se = np.sqrt(diag_vcov)
                return vcov, se, False
            else:
                print_warning("Optimizer's Hessian inverse is not positive definite.")
        except Exception as e:
            print_warning(f"Could not use optimizer's Hessian inverse: {e}")

    print_warning("Falling back to numerical Hessian calculation for SEs.")
    
    # 2. Numerical Hessian calculation
    numerical_hess = None
    try:
        step = np.finfo(float).eps**(1/3)
        numerical_hess = approx_fprime(final_params, lambda p: approx_fprime(p, neg_loglik_fun, step), step)
        
        if not np.all(np.isfinite(numerical_hess)):
            print_warning("Numerical Hessian contains non-finite values.")
            raise np.linalg.LinAlgError("Non-finite Hessian")

        numerical_hess = (numerical_hess + numerical_hess.T) / 2
        
        cond_num = np.linalg.cond(numerical_hess)
        if cond_num > 1e12:
            print_warning(f"Hessian is poorly conditioned (cond={cond_num:.2e})")
        
        vcov = np.linalg.inv(numerical_hess)
        diag_vcov = np.diag(vcov)
        if np.all(diag_vcov > 0):
            se = np.sqrt(diag_vcov)
            return vcov, se, False
        else:
             print_warning("Inverted numerical Hessian is not positive definite.")
    except np.linalg.LinAlgError:
        print_warning("Numerical Hessian is singular and cannot be inverted.")
    except Exception as e:
        print_warning(f"Failed to calculate numerical Hessian inverse: {e}")

    # 3. Try pseudo-inverse of numerical Hessian
    if numerical_hess is not None and np.all(np.isfinite(numerical_hess)):
        print_warning("Trying pseudo-inverse of the Hessian.")
        try:
            vcov_pinv = np.linalg.pinv(numerical_hess)
            diag_vcov_pinv = np.diag(vcov_pinv)
            if np.all(diag_vcov_pinv >= 0):
                se = np.sqrt(np.maximum(0, diag_vcov_pinv))
                return vcov_pinv, se, False
            else:
                print_warning("Pseudo-inverse resulted in negative diagonal elements.")
        except Exception as e:
            print_warning(f"Pseudo-inverse calculation failed: {e}")

    # 4. Final fallback: Hessian "bending" via eigenvalue adjustment
    if numerical_hess is not None and np.all(np.isfinite(numerical_hess)):
        print_warning("Trying Hessian bending as a last resort.")
        try:
            numerical_hess = (numerical_hess + numerical_hess.T) / 2
            eigenvalues, V = np.linalg.eigh(numerical_hess)
            non_pos_count = np.sum(eigenvalues <= 1e-6)
            if non_pos_count > 0:
                 print_warning(f"Bending Hessian: {non_pos_count}/{num_params} eigenvalues are non-positive.")
            
            min_eig = 1e-6
            eigenvalues[eigenvalues <= min_eig] = min_eig
            H_new = V @ np.diag(eigenvalues) @ V.T
            vcov = np.linalg.inv(H_new)
            diag_vcov = np.diag(vcov)
            if np.all(diag_vcov >= 0):
                se = np.sqrt(np.maximum(0, diag_vcov))
                return vcov, se, True
        except Exception as e:
            print_warning(f"Hessian bending failed: {e}")
            
    # If all methods fail, return NaNs
    print_warning("All reliable methods to calculate standard errors failed. Returning NaNs.")
    vcov = np.full((num_params, num_params), np.nan)
    se = np.full(num_params, np.nan)
    return vcov, se, False

def _robust_optimization(loglik_fn, k, initial_guess=None, display=False):
    """
    Robust optimization with multiple methods and random starts.
    """
    if initial_guess is None:
        initial_guess = np.zeros(k)
    
    best_fval = np.inf
    best_params = None
    best_result = None
    
    optimizers_to_try = ['BFGS', 'L-BFGS-B']
    n_random_starts = 3

    for method in optimizers_to_try:
        for i in range(n_random_starts):
            if i == 0:
                current_theta0 = initial_guess.copy()
            else:
                current_theta0 = initial_guess + np.random.standard_normal(k) * 0.1
            
            try:
                if method == 'L-BFGS-B':
                    bounds = [(None, None)] * k
                    # Bound variance parameters to avoid extreme values
                    for j in range(3, k):  # variance parameters start at index 3
                        bounds[j] = (-50, 50)  # log-variance bounds
                    
                    res = minimize(
                        loglik_fn,
                        x0=current_theta0,
                        method=method,
                        bounds=bounds,
                        options={'disp': display, 'maxiter': 5000}
                    )
                else:
                    res = minimize(
                        loglik_fn,
                        x0=current_theta0,
                        method=method,
                        options={'disp': display, 'maxiter': 5000}
                    )
            except Exception as e:
                if display:
                    print(f"Optimizer {method} failed with start {i}: {e}")
                continue

            if res.success and res.fun < best_fval:
                best_fval = res.fun
                best_params = res.x
                best_result = res

        if best_result is not None and best_result.success:
            break

    return best_result

def tre_estimator(Tdata, ny, nf, nu, nv, **kwargs):
    """
    True Random Effects Stochastic Frontier Model Estimation.
    
    The TRE model includes random firm-specific frontier effects:
    y_it = β'x_it + α_i + v_it - u_it
    where α_i ~ N(0, σ_α²) are random frontier effects
    
    Parameters:
    -----------
    Tdata : pandas.DataFrame
        Panel data with columns: y, xf1, xf2, xu, xv, id, ti
    ny : str or int
        Dependent variable column name/index
    nf : list
        Frontier variable column names/indices
    nu : list  
        Inefficiency variance variable column names/indices
    nv : list
        Noise variance variable column names/indices
    **kwargs : dict
        Additional parameters including:
        - s: 1 for production frontier (default), -1 for cost frontier
        - ns: number of Halton draws for simulation (default 200)
        - rep_num: replication number for tracking
        - max_exp: maximum exponent to prevent overflow
        - verbose: whether to print estimation results (default True)
        - cost: if True, estimate cost frontier (s=-1)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - converged: bool, whether optimization converged
        - loglik: float, log-likelihood value
        - bhat: array, estimated parameters
        - se: array, standard errors
        - pval: array, p-values
        - effhat: array, predicted efficiency scores
        - bf: frontier parameters [estimates, std errors, p-values]
        - bu: inefficiency variance parameters [estimates, std errors, p-values]
        - bv: noise variance parameters [estimates, std errors, p-values]
        - ba: random effect variance parameter [estimate, std error, p-value]
    """
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Parse arguments
    s = 1  # Default: production frontier
    if kwargs.get('cost', False):
        s = -1
    s = kwargs.get('s', s)
    
    ns = kwargs.get('ns', 200)  # Number of simulation draws
    rep_num = kwargs.get('rep_num', 'N/A')
    max_exp = kwargs.get('max_exp', 50)
    verbose = kwargs.get('verbose', True)
    
    # Extract data
    y = Tdata['y'].values
    xf1 = Tdata['xf1'].values
    xf2 = Tdata['xf2'].values
    xu = Tdata['xu'].values
    xv = Tdata['xv'].values
    id_vals = Tdata['id'].values
    
    n = len(y)
    ni = Tdata['id'].nunique()  # Number of firms
    
    # Design matrices (add intercepts)
    Xf = np.column_stack([xf1, xf2, np.ones(n)])    # Frontier: [xf1, xf2, constant]
    Xu = np.column_stack([xu, np.ones(n)])          # Inefficiency variance: [xu, constant]
    Xv = np.column_stack([xv, np.ones(n)])          # Noise variance: [xv, constant]
    
    # Parameter dimensions
    kf = Xf.shape[1]
    ku = Xu.shape[1]
    kv = Xv.shape[1]
    k = kf + ku + kv + 1  # +1 for log(σ_α²)
    
    # Generate Halton draws for random effects
    try:
        sampler = qmc.Halton(d=1, scramble=True)
        a0 = sampler.random(n=ni * ns)
        A = norm.ppf(a0).reshape(ni, ns)  # Standard normal draws
    except:
        # Fallback to regular random draws
        A = np.random.randn(ni, ns)
    
    # Create mapping from observation to firm index
    unique_ids = np.sort(Tdata['id'].unique())
    id_map = {firm_id: i for i, firm_id in enumerate(unique_ids)}
    firm_indices = np.array([id_map[i] for i in id_vals])
    
    def loglike_tre(params):
        """Negative log-likelihood function for TRE model with simulation."""
        try:
            # Extract parameters
            bf = params[:kf]
            bu = params[kf:kf+ku]
            bv = params[kf+ku:kf+ku+kv]
            log_sigma_alpha = params[kf+ku+kv]
            
            # Transform variance parameter
            log_sigma_alpha = np.clip(log_sigma_alpha, -max_exp, max_exp)
            sigma_alpha = np.exp(log_sigma_alpha)
            
            # Linear predictors for variances
            log_sigu = Xu @ bu
            log_sigv = Xv @ bv
            log_sigu = np.clip(log_sigu, -max_exp, max_exp)
            log_sigv = np.clip(log_sigv, -max_exp, max_exp)
            
            # Variances with safety
            epsilon = 1e-9
            sigu = np.exp(log_sigu)
            sigv = np.exp(log_sigv)
            
            # Random effects scaled by σ_α
            alpha_draws = sigma_alpha * A  # Shape: (ni, ns)
            
            # Map random effects to observations
            alpha_obs = alpha_draws[firm_indices, :]  # Shape: (n, ns)
            
            # Frontier equation with random effects
            xb = (Xf @ bf).reshape(-1, 1)  # Shape: (n, 1)
            
            # Residuals for each simulation draw
            e = y.reshape(-1, 1) - xb - alpha_obs  # Shape: (n, ns)
            
            # Variance components
            sig = np.sqrt(sigu.reshape(-1, 1)**2 + sigv.reshape(-1, 1)**2 + epsilon)  # Shape: (n, 1)
            lam = (sigu / (sigv + epsilon)).reshape(-1, 1)  # Shape: (n, 1)
            
            # Log-density for each draw
            z = -s * e * lam / sig
            
            # Prevent numerical issues
            z = np.clip(z, -10, 10)  # Avoid extreme values
            
            log_phi = norm.logpdf(e / sig)
            log_Phi = norm.logcdf(z)
            
            # Ensure numerical stability
            log_Phi = np.maximum(log_Phi, -50)  # Avoid log(0)
            
            # Log-density: log(2) + log(φ(e/σ)) + log(Φ(z)) - log(σ)
            log_f = np.log(2) + log_phi + log_Phi - np.log(sig)
            
            # Sum log-densities over time for each firm and simulation
            log_likelihood_firm_sim = np.zeros((ni, ns))
            for i in range(n):
                firm_idx = firm_indices[i]
                log_likelihood_firm_sim[firm_idx, :] += log_f[i, :]
            
            # Use log-sum-exp trick for numerical stability
            max_ll = np.max(log_likelihood_firm_sim, axis=1, keepdims=True)
            log_integral = max_ll.flatten() + np.log(np.mean(np.exp(log_likelihood_firm_sim - max_ll), axis=1))
            
            # Total log-likelihood
            ll = np.sum(log_integral)
            
            return -ll if np.isfinite(ll) else 1e15
            
        except Exception as e:
            return 1e15
    
    try:
        # Improved initial parameter guess
        beta00 = np.zeros(k)
        
        # OLS for frontier parameters
        try:
            XtX = Xf.T @ Xf
            if np.linalg.cond(XtX) > 1e12:
                lambda_reg = 0.01
                beta00[:kf] = np.linalg.solve(XtX + lambda_reg * np.eye(kf), Xf.T @ y)
            else:
                beta00[:kf] = np.linalg.solve(XtX, Xf.T @ y)
        except:
            beta00[:kf] = np.linalg.lstsq(Xf, y, rcond=None)[0]
        
        # Initial values for variance components based on residuals
        res0 = y - Xf @ beta00[:kf]
        var_res0 = np.var(res0)
        
        # Conservative initial values
        beta00[kf:kf+ku] = [np.log(var_res0 / 2 + 1e-6), -1.0]  # bu parameters
        beta00[kf+ku:kf+ku+kv] = [np.log(var_res0 / 2 + 1e-6), -1.0]  # bv parameters
        beta00[kf+ku+kv] = np.log(0.1)  # log(σ_α²) - small random effect variance
        
        # Robust optimization
        logger = IterationLogger('TRE', rep_num)
        result = _robust_optimization(loglike_tre, k, initial_guess=beta00, display=verbose)
        
        if result is None or not result.success:
            if verbose:
                print("⚠️  TRE Estimation Failed - Optimization did not converge")
            return {
                'converged': False,
                'loglik': np.nan,
                'bhat': np.full(k, np.nan),
                'se': np.full(k, np.nan),
                'pval': np.full(k, np.nan),
                'effhat': np.full(n, np.nan),
                'hessian': np.full((k, k), np.nan),
                'bf': np.full((kf, 3), np.nan),
                'bu': np.full((ku, 3), np.nan),
                'bv': np.full((kv, 3), np.nan),
                'ba': np.full((1, 3), np.nan)
            }
        
        # Extract results
        bhat = result.x
        loglik = -result.fun
        
        # Robust standard error calculation
        try:
            hess_inv_approx = getattr(result, 'hess_inv', None)
            hessian, se, bending_used = _calculate_vcov_safe(
                hess_inv_approx, 
                loglike_tre, 
                bhat, 
                display=verbose
            )
            
            if bending_used and verbose:
                print("ℹ️  Used Hessian bending for positive definiteness")
                
        except Exception as e:
            if verbose:
                print(f"⚠️  Warning: Standard error computation failed: {e}")
            hessian = np.eye(k)
            se = np.full(k, np.nan)
        
        # Compute p-values using safe calculation
        param_names = ['xf1', 'xf2', 'const_f', 'xu', 'const_u', 'xv', 'const_v', 'log_sigma_alpha']
        pval = np.array([
            _safe_pvalue_calculation(bhat[i], se[i], param_names[i], display=False) 
            for i in range(k)
        ])
        
        # Organize results by parameter type
        bf_results = np.column_stack([bhat[:kf], se[:kf], pval[:kf]])
        bu_results = np.column_stack([bhat[kf:kf+ku], se[kf:kf+ku], pval[kf:kf+ku]])
        bv_results = np.column_stack([bhat[kf+ku:kf+ku+kv], se[kf+ku:kf+ku+kv], pval[kf+ku:kf+ku+kv]])
        ba_results = np.column_stack([bhat[kf+ku+kv:], se[kf+ku+kv:], pval[kf+ku+kv:]]).reshape(1, -1)
        
        # Compute efficiency scores using Jondrow et al. formula with random effects
        try:
            bf = bhat[:kf]
            bu = bhat[kf:kf+ku]
            bv = bhat[kf+ku:kf+ku+kv]
            log_sigma_alpha = bhat[kf+ku+kv]
            
            # Recalculate components with safety
            sigma_alpha = np.exp(np.clip(log_sigma_alpha, -max_exp, max_exp))
            sigu = np.exp(np.clip(Xu @ bu, -max_exp, max_exp))
            sigv = np.exp(np.clip(Xv @ bv, -max_exp, max_exp))
            
            # Add small epsilon for numerical stability
            epsilon = 1e-9
            sigu = np.maximum(sigu, epsilon)
            sigv = np.maximum(sigv, epsilon)
            
            # Use posterior mean of random effects (set to 0 for efficiency calculation)
            xb = Xf @ bf
            
            # Residuals without random effects (conservative approach)
            e = y - xb
            
            # Jondrow et al. formula for inefficiency
            sigma_sq = sigu**2 + sigv**2
            mu_star = -e * (sigu**2 / sigma_sq)
            sigma_star = np.sqrt((sigu**2 * sigv**2) / sigma_sq)
            
            # Ratio of PDF to CDF with safety
            Z = mu_star / np.maximum(sigma_star, epsilon)
            pdf_val = norm.pdf(Z)
            cdf_val = norm.cdf(Z)
            cdf_val = np.clip(cdf_val, 1e-12, 1 - 1e-12)
            phi_ratio = pdf_val / cdf_val

            # Conditional mean of u
            E_u_eps = mu_star + sigma_star * phi_ratio
            
            # Technical efficiency
            effhat = np.exp(-np.maximum(E_u_eps, 0))
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Warning: Efficiency computation failed: {e}")
            effhat = np.full(n, np.nan)
        
        # Print results if verbose
        if verbose:
            print_tre_results(bf_results, bu_results, bv_results, ba_results, loglik, s, effhat)
        
        return {
            'converged': True,
            'loglik': loglik,
            'bhat': bhat,
            'se': se,
            'pval': pval,
            'effhat': effhat,
            'hessian': hessian,
            'bf': bf_results,
            'bu': bu_results,
            'bv': bv_results,
            'ba': ba_results
        }
        
    except Exception as e:
        if verbose:
            print(f"❌ TRE estimation failed: {e}")
        return {
            'converged': False,
            'loglik': np.nan,
            'bhat': np.full(k, np.nan),
            'se': np.full(k, np.nan),
            'pval': np.full(k, np.nan),
            'effhat': np.full(n, np.nan),
            'hessian': np.full((k, k), np.nan),
            'bf': np.full((kf, 3), np.nan),
            'bu': np.full((ku, 3), np.nan),
            'bv': np.full((kv, 3), np.nan),
            'ba': np.full((1, 3), np.nan)
        }

def print_tre_results(bf, bu, bv, ba, loglik, s, effhat):
    """Print nicely formatted TRE estimation results."""
    print("\n" + "="*80)
    print("TRUE RANDOM EFFECTS STOCHASTIC FRONTIER MODEL ESTIMATION RESULTS")
    print("="*80)
    
    # Model type
    model_type = "Production Frontier" if s == 1 else "Cost Frontier"
    print(f"Model Type: {model_type}")
    print(f"Log-likelihood: {loglik:.6f}")
    
    # Efficiency statistics
    if not np.any(np.isnan(effhat)):
        print(f"Mean Efficiency: {np.mean(effhat):.4f}")
        print(f"Std Efficiency:  {np.std(effhat):.4f}")
        print(f"Min Efficiency:  {np.min(effhat):.4f}")
        print(f"Max Efficiency:  {np.max(effhat):.4f}")
    
    print("\n" + "-"*80)
    print("PARAMETER ESTIMATES")
    print("-"*80)
    
    # Frontier parameters
    print("\nFrontier Parameters:")
    print("Variable      Coefficient   Std. Error    t-statistic   P-value")
    print("-" * 65)
    var_names = ['xf1', 'xf2', 'constant']
    for i, name in enumerate(var_names):
        coef = bf[i, 0]
        se = bf[i, 1]
        pval = bf[i, 2]
        t_stat = coef / se if se > 0 else np.nan
        stars = get_significance_stars(pval)
        print(f"{name:<12} {coef:>10.6f}   {se:>10.6f}   {t_stat:>10.3f}   {pval:>8.4f}{stars}")
    
    # Inefficiency variance parameters
    print("\nInefficiency Variance Parameters:")
    print("Variable      Coefficient   Std. Error    t-statistic   P-value")
    print("-" * 65)
    var_names = ['xu', 'constant']
    for i, name in enumerate(var_names):
        coef = bu[i, 0]
        se = bu[i, 1]
        pval = bu[i, 2]
        t_stat = coef / se if se > 0 else np.nan
        stars = get_significance_stars(pval)
        print(f"{name:<12} {coef:>10.6f}   {se:>10.6f}   {t_stat:>10.3f}   {pval:>8.4f}{stars}")
    
    # Noise variance parameters
    print("\nNoise Variance Parameters:")
    print("Variable      Coefficient   Std. Error    t-statistic   P-value")
    print("-" * 65)
    var_names = ['xv', 'constant']
    for i, name in enumerate(var_names):
        coef = bv[i, 0]
        se = bv[i, 1]
        pval = bv[i, 2]
        t_stat = coef / se if se > 0 else np.nan
        stars = get_significance_stars(pval)
        print(f"{name:<12} {coef:>10.6f}   {se:>10.6f}   {t_stat:>10.3f}   {pval:>8.4f}{stars}")
    
    # Random effect variance parameter
    print("\nRandom Effect Variance Parameter:")
    print("Variable      Coefficient   Std. Error    t-statistic   P-value")
    print("-" * 65)
    coef = ba[0, 0]
    se = ba[0, 1]
    pval = ba[0, 2]
    t_stat = coef / se if se > 0 else np.nan
    stars = get_significance_stars(pval)
    print(f"{'log(σ_α²)':<12} {coef:>10.6f}   {se:>10.6f}   {t_stat:>10.3f}   {pval:>8.4f}{stars}")
    
    print("\nSignificance levels: *** 1%, ** 5%, * 10%")
    print("="*80)

def get_significance_stars(pval):
    """Return significance stars based on p-value."""
    if np.isnan(pval):
        return ""
    elif pval < 0.01:
        return " ***"
    elif pval < 0.05:
        return " **"
    elif pval < 0.10:
        return " *"
    else:
        return "" 
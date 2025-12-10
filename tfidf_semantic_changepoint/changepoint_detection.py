"""
Bayesian changepoint detection using PyMC
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az


def fit_changepoint_model(drift_df, draws=2000, tune=2000, chains=4, random_seed=42):
    """
    Fit Bayesian changepoint model to semantic drift data.
    
    Args:
        drift_df: DataFrame with 'year' and 'drift' columns
        draws: Number of posterior samples
        tune: Number of tuning samples
        chains: Number of MCMC chains
        random_seed: Random seed for reproducibility
        
    Returns:
        PyMC trace object
    """
    years = drift_df["year"].values
    y = drift_df["drift"].values
    
    year_min, year_max = years.min(), years.max()
    
    with pm.Model() as cp_model:
        tau = pm.DiscreteUniform("tau", lower=year_min, upper=year_max)
        
        mu_pre = pm.Normal("mu_pre", mu=0.0, sigma=1.0)
        mu_post = pm.Normal("mu_post", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        
        mu_t = pm.math.switch(years < tau, mu_pre, mu_post)
        
        y_obs = pm.Normal("y_obs", mu=mu_t, sigma=sigma, observed=y)
        
        cp_trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.9,
            chains=chains,
            random_seed=random_seed,
        )
    
    return cp_trace


def plot_changepoint_posterior(cp_trace, title="Posterior distribution of change-point year (tau)", save_path=None):
    """
    Plot posterior distribution of changepoint year.
    
    Args:
        cp_trace: PyMC trace object
        title: Plot title
        save_path: Optional path to save plot
    """
    tau_vals = cp_trace.posterior['tau'].values.flatten()
    
    year_min = int(tau_vals.min())
    year_max = int(tau_vals.max())
    
    plt.figure(figsize=(10, 6))
    plt.hist(tau_vals, bins=np.arange(year_min, year_max + 2) - 0.5, edgecolor='black')
    plt.xticks(range(year_min, year_max + 1))
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Posterior count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def analyze_division_changepoint(df, division, baseline_years=None, target_year=2025, save_path=None):
    """
    Perform changepoint analysis for a specific division.
    
    Args:
        df: DataFrame with embeddings
        division: Division name
        baseline_years: Years to use as baseline
        target_year: Year to compare against baseline
        save_path: Optional path to save plot
        
    Returns:
        Tuple of (drift_df, cp_trace)
    """
    df_sub = df[df["division_name"] == division].copy()
    
    drift_df = compute_semantic_drift(df_sub, baseline_years, target_year)
    cp_trace = fit_changepoint_model(drift_df)
    
    title = f"Posterior distribution of change-point year (tau)\n({division})"
    plot_changepoint_posterior(cp_trace, title=title, save_path=save_path)
    
    return drift_df, cp_trace


def analyze_multiple_divisions(df, divisions, baseline_years=None, target_year=2025, save_dir=None):
    """
    Perform changepoint analysis for multiple divisions (pooled).
    
    Args:
        df: DataFrame with embeddings
        divisions: List of division names
        baseline_years: Years to use as baseline
        target_year: Year to compare against baseline
        save_dir: Optional directory to save plots
        
    Returns:
        Tuple of (drift_df, cp_trace)
    """
    df_sub = df[df["division_name"].isin(divisions)].copy()
    
    drift_df = compute_semantic_drift(df_sub, baseline_years, target_year)
    cp_trace = fit_changepoint_model(drift_df)
    
    divisions_str = ", ".join(divisions) if len(divisions) <= 3 else f"{len(divisions)} selected programs"
    title = f"Posterior distribution of change-point year (tau)\n({divisions_str} pooled)"
    
    save_path = None
    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / "changepoint_posterior.png"
    
    plot_changepoint_posterior(cp_trace, title=title, save_path=save_path)
    
    return drift_df, cp_trace


from .semantic_similarity import compute_semantic_drift


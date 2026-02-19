"""Utility helpers for pooling and small helpers."""
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats


def rubins_rules_from_models(models: List) -> Dict[str, Any]:
    """Pool coefficients and variances from fitted models that expose
    .params_ (pandas Series) and .variance_matrix_ (pandas DataFrame).
    Returns a dict with pooled table (pandas.DataFrame) and matrices."""

    m = len(models)
    betas = []
    vars_ = []
    for mod in models:
        b = mod.params_.copy()
        V = mod.variance_matrix_.loc[b.index, b.index].values
        betas.append(b.values)
        vars_.append(V)
    betas = np.vstack(betas)
    beta_bar = betas.mean(axis=0)
    W = np.mean(np.stack(vars_), axis=0)
    B = np.cov(betas, rowvar=False, ddof=1) if m > 1 else np.zeros_like(W)
    T = W + (1 + 1.0/m) * B
    se = np.sqrt(np.diag(T))
    z = beta_bar / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    index = models[0].params_.index
    pooled = pd.DataFrame({'coef': beta_bar, 'se': se, 'z': z, 'p': pvals, 'var': np.diag(T)}, index=index)
    return {'pooled_table': pooled, 'W': W, 'B': B, 'T': T, 'm': m}
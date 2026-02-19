"""
Core modeling functions: prepare_gap_time, fit_stratified_cox, fit_rec_ev_model
This file contains the primary modeling API used by the package.
"""
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from .utils import rubins_rules_from_models


def prepare_gap_time(df: pd.DataFrame,
                     id_col: str,
                     time_col: str,
                     event_col: str,
                     episode_col: str,
                     start_col: Optional[str] = None,
                     make_gap: bool = True) -> pd.DataFrame:
    """
    Compute or validate start/stop columns for gap-time Cox modeling.
    """
    dfc = df.copy()
    if start_col is None and make_gap:
        dfc = dfc.sort_values([id_col, time_col]).reset_index(drop=True)
        dfc["start"] = dfc.groupby(id_col)[time_col].shift(1).fillna(0.0).astype(float)
        dfc["stop"] = dfc[time_col].astype(float)
    else:
        if start_col is None:
            raise ValueError("Either provide start_col or allow make_gap=True.")
        dfc["start"] = dfc[start_col].astype(float)
        dfc["stop"] = dfc[time_col].astype(float)

    dfc["event"] = dfc[event_col].astype(int)
    return dfc


def fit_stratified_cox(df: pd.DataFrame,
                       covariates: List[str],
                       id_col: str,
                       episode_col: str,
                       start_col: Optional[str],
                       stop_col: str,
                       event_col: str,
                       robust: bool = True,
                       strata_col: Optional[str] = None,
                       penalizer: float = 0.0,
                       **lifelines_kwargs) -> CoxPHFitter:
    """
    Fit a stratified Cox model using lifelines.CoxPHFitter.
    Supports left truncation via entry_col.
    """
    cph = CoxPHFitter(penalizer=penalizer)
    df_fit = df.copy()

    # Required columns
    needed = [stop_col, event_col] + covariates + [id_col]
    if episode_col is not None:
        needed.append(episode_col)
    if start_col:
        needed.insert(0, start_col)

    missing = [c for c in needed if c not in df_fit.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric
    if start_col:
        df_fit[start_col] = pd.to_numeric(df_fit[start_col])
    df_fit[stop_col] = pd.to_numeric(df_fit[stop_col])
    df_fit[event_col] = df_fit[event_col].astype(int)

    # Lifelines fit arguments
    fit_args: Dict[str, Any] = {
        "df": df_fit,
        "duration_col": stop_col,
        "event_col": event_col,
    }

    # Left truncation → use entry_col instead of start_col
    if start_col:
        fit_args["entry_col"] = start_col

    # Robust sandwich variance via cluster_col
    if robust:
        fit_args["cluster_col"] = id_col

    # Strata
    '''strata = strata_col if strata_col is not None else episode_col
    fit_args["strata"] = [strata]'''
    if strata_col is not None:
        fit_args["strata"] = [strata_col]
    elif episode_col is not None:
        fit_args["strata"] = [episode_col]
    # else: no stratification

    fit_args.update(lifelines_kwargs)

    cph.fit(**fit_args)
    return cph


def fit_rec_ev_model(imputed_dfs: List[pd.DataFrame],
                     covariates: List[str],
                     id_col: str,
                     time_col: str,
                     event_col: str,
                     episode_col: str,
                     gap_time: bool = True,
                     start_col: Optional[str] = None,
                     robust: bool = True,
                     strata_col: Optional[str] = None,
                     penalizer: float = 0.0,
                     lifelines_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Fit episode-stratified Cox recurrent event model across multiple imputations.
    """
    if lifelines_kwargs is None:
        lifelines_kwargs = {}

    fitted_models = []

    for df in imputed_dfs:
        if gap_time:
            df_prep = prepare_gap_time(
                df,
                id_col=id_col,
                time_col=time_col,
                event_col=event_col,
                episode_col=None,
                start_col=start_col,
                make_gap=(start_col is None)
            )
            s_col = "start"
            t_col = "stop"
            e_col = "event"
        else:
            df_prep = df.copy()
            df_prep[time_col] = pd.to_numeric(df_prep[time_col])
            s_col = None
            t_col = time_col
            e_col = event_col

        model = fit_stratified_cox(
            df_prep,
            covariates=covariates,
            id_col=id_col,
            episode_col=None,
            start_col=s_col,
            stop_col=t_col,
            event_col=e_col,
            robust=robust,
            strata_col=strata_col,
            penalizer=penalizer,
            **lifelines_kwargs
        )
        fitted_models.append(model)

    pooled = rubins_rules_from_models(fitted_models)
    return {"models": fitted_models, "pooled": pooled}

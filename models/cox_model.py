# models/cox_model.py

from src.mirecsurv_py.core import fit_stratified_cox
from .base_model import BaseSurvivalModel

class CoxModel(BaseSurvivalModel):

    def fit(self, df):
        self.model = fit_stratified_cox(
            df=df,
            covariates=self.features,
            id_col="name",
            episode_col="episode_col",
            start_col="entry",
            stop_col="dur",
            event_col="event"
        )

    def predict_survival(self, df, times):
        surv = self.model.predict_survival_function(df, times=times)
        return surv.T

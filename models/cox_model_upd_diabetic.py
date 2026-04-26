import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.mirecsurv_py.core import fit_stratified_cox
from models.base_model import BaseSurvivalModel 

class CoxModelUpd(BaseSurvivalModel):

    def __init__(
        self,
        features,
        mode="recurrent",   # so this would be "recurrent" or "independent"
        use_episode=False,
        **params
    ):
        super().__init__(features)
        self.mode = mode
        self.use_episode = use_episode
        self.params = params


    def fit(self, df):
        if self.mode == "recurrent":
            self.model = fit_stratified_cox(
                df=df,
                covariates=self.features,
                id_col="patient_id",
                episode_col="episode_col" if self.use_episode else None,
                start_col="start",
                stop_col="stop",
                event_col="event",
                **self.params
            )

        elif self.mode == "independent":
            self.model = fit_stratified_cox(
                df=df,
                covariates=self.features,
                id_col=None,
                episode_col=None,
                start_col=None,
                stop_col="time",
                event_col="event",
                robust=False,
                **self.params
            )

        else:
            raise ValueError("mode must be 'recurrent' or 'independent'")
        self.model.print_summary()
        print(type(self.model))


    def predict_survival(self, df, times):
        surv = self.model.predict_survival_function(df, times=times)
        return surv.T
    
    def predict_cumulative_hazard(self, df, times):
        ch = self.model.predict_cumulative_hazard(df, times=times)
        return ch


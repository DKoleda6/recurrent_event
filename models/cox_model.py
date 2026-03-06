from src.mirecsurv_py.core import fit_stratified_cox
from .base_model import BaseSurvivalModel

class CoxModel(BaseSurvivalModel):

    def fit(self, df):
        self.model = fit_stratified_cox(
            df=df,
            covariates=self.features,
            id_col="name",
            episode_col=None,
            start_col="start",
            stop_col="stop",
            event_col="event"
        )
        self.model.print_summary()
        print(type(self.model))
        '''summary = self.model.summary
        print(summary)'''



    def predict_survival(self, df, times):
        surv = self.model.predict_survival_function(df, times=times)
        return surv.T
    
    def predict_cumulative_hazard(self, df, times):
        ch = self.model.predict_cumulative_hazard(df, times=times)
        return ch


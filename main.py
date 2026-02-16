import numpy as np

from data.data_processor import DataProcessor
from data.splitter import SurvivalSplitter

from models.cox_model import CoxModel

from metrics.ibs import IBSMetric
from metrics.ibs_remain import IBSRemainMetric
from metrics.auprc import AUPRCMetric

from pipeline.evaluator import SurvivalEvaluator

from survivors.constants import get_y


# 1. Load Data
processor = DataProcessor("data/individual_custody_timeline_rfm.csv")
cox_df = processor.load_and_prepare()

# 2. Train/Test Split
splitter = SurvivalSplitter(test_size=0.2)
train_df, test_df = splitter.split_by_individual(cox_df)

print(f"Train individuals: {train_df['name'].nunique()}")
print(f"Test individuals: {test_df['name'].nunique()}")

# 3. Prepare Survival Targets
train_df["time"] = train_df["entry"] + train_df["dur"]
test_df["time"] = test_df["entry"] + test_df["dur"]

train_cens = ~train_df["event"].astype(bool)
test_cens = ~test_df["event"].astype(bool)

survival_train = get_y(
    cens=train_cens,
    time=train_df["time"],
    competing=False
)

survival_test = get_y(
    cens=test_cens,
    time=test_df["time"],
    competing=False
)

# 4. Train Model
model = CoxModel(features=["age"])

model.fit(train_df)

# 5. Predict on TEST
times = np.linspace(0, train_df["dur"].max(), 200)

predictions = model.predict_survival(test_df, times)
estimate = predictions.values

# 6. Compute Metrics
ibs_metric = IBSMetric()
mean_ibs, ibs_by_time = ibs_metric.compute(
    survival_train,
    survival_test,
    estimate,
    times
)

ibs_remain_metric = IBSRemainMetric()
ibs_remain = ibs_remain_metric.compute(
    survival_train,
    survival_test,
    estimate,
    times
)

auprc_metric = AUPRCMetric()
auprc = auprc_metric.compute(
    survival_train,
    survival_test,
    estimate,
    times
)

# 7. Evaluate and Save
evaluator = SurvivalEvaluator()

evaluator.evaluate_and_save(
    predictions,
    times,
    mean_ibs,
    ibs_by_time,
    ibs_remain,
    auprc
)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SurvivalEvaluator:

    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/csv", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)

    def evaluate_and_save(
        self,
        predictions,
        times,
        mean_ibs,
        ibs_by_time,
        ibs_remain,
        auprc
    ):

        # Per-individual metrics
        individual_df = pd.DataFrame({
            "individual_id": predictions.index,
            "ibs_remain": ibs_remain,
            "auprc": auprc
        })

        individual_df.to_csv(
            f"{self.output_dir}/csv/individual_metrics.csv",
            index=False
        )

        # IBS over time
        ibs_time_df = pd.DataFrame({
            "time": times,
            "ibs_error": ibs_by_time
        })

        ibs_time_df.to_csv(
            f"{self.output_dir}/csv/ibs_by_time.csv",
            index=False
        )

        # Summary metrics
        summary_df = pd.DataFrame({
            "metric": ["Mean IBS", "Mean AUPRC"],
            "value": [mean_ibs, np.mean(auprc)]
        })

        summary_df.to_csv(
            f"{self.output_dir}/csv/model_summary.csv",
            index=False
        )

        # Plot IBS
        plt.figure(figsize=(10, 6))
        plt.plot(times, ibs_by_time)
        plt.axhline(y=mean_ibs, linestyle="--")
        plt.title("Integrated Brier Score Over Time")
        plt.xlabel("Time")
        plt.ylabel("IBS")
        plt.grid(True)

        plt.savefig(
            f"{self.output_dir}/plots/ibs_time_plot.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print("Evaluation complete. Results saved to reports/.")

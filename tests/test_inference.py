import unittest

import numpy as np
import pandas as pd

from src.inference import (
    extract_time_series,
    predict_batch,
    summarize_predictions,
)


class FakeClassifier:
    classes_ = np.array(["ARIMA", "Prophet"])
    n_features_in_ = 10

    def predict(self, features):
        return np.array(["ARIMA" if value >= 0 else "Prophet" for value in features["mean"]])

    def predict_proba(self, features):
        return np.tile([0.75, 0.25], (len(features), 1))


class InferenceTests(unittest.TestCase):
    def test_extracts_each_row_and_drops_missing_observations(self):
        frame = pd.DataFrame(
            {
                "Series ID": ["first", "second"],
                "t1": [1.0, 3.0],
                "t2": [None, 4.0],
                "t3": [2.0, np.inf],
            }
        )

        series = extract_time_series(frame)

        self.assertEqual([item.series_id for item in series], ["first", "second"])
        self.assertEqual(series[0].values.tolist(), [1.0, 2.0])
        self.assertEqual(series[0].removed_count, 1)
        self.assertEqual(series[1].values.tolist(), [3.0, 4.0])
        self.assertEqual(series[1].removed_count, 1)

    def test_rejects_non_numeric_observation_values(self):
        frame = pd.DataFrame({"ID": ["series"], "t1": ["not numeric"]})

        with self.assertRaisesRegex(ValueError, "non-numeric values"):
            extract_time_series(frame)

    def test_uses_the_first_non_numeric_column_as_the_identifier(self):
        frame = pd.DataFrame(
            {
                "t1": [2.0],
                "Series ID": ["series"],
                "t2": [3.0],
            }
        )

        series = extract_time_series(frame)

        self.assertEqual(series[0].series_id, "series")
        self.assertEqual(series[0].values.tolist(), [2.0, 3.0])

    def test_predicts_short_series_and_reports_empty_rows(self):
        series = extract_time_series(
            pd.DataFrame(
                {
                    "ID": ["short", "empty", "another"],
                    "t1": [1.0, None, 2.0],
                    "t2": [None, None, 3.0],
                }
            )
        )
        progress = []

        batch = predict_batch(
            series,
            FakeClassifier(),
            [
                "length",
                "mean",
                "variance",
                "standard_deviation",
                "trend",
                "autocorrelation",
                "skewness",
                "kurtosis",
                "seasonality_strength",
                "adf_statistic",
            ],
            progress_callback=lambda completed, total: progress.append(
                (completed, total)
            ),
        )

        self.assertEqual(batch.recommendations["Series ID"].tolist(), ["short", "another"])
        self.assertEqual(batch.recommendations.columns.tolist(), [
            "Series ID",
            "Recommended Model",
            "Confidence",
        ])
        self.assertEqual(batch.details[0].recommendation, "ARIMA")
        self.assertEqual(batch.errors, {1: "No valid numeric observations."})
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])

    def test_summarizes_aggregate_predictions_and_ties(self):
        majority = summarize_predictions(
            pd.DataFrame(
                {
                    "Recommended Model": ["ARIMA", "ARIMA", "Prophet"],
                    "Confidence": [0.8, 0.6, 0.9],
                }
            )
        )
        tie = summarize_predictions(
            pd.DataFrame(
                {
                    "Recommended Model": ["ARIMA", "Prophet"],
                    "Confidence": [0.8, 0.7],
                }
            )
        )

        self.assertEqual(majority.overall_model, "ARIMA")
        self.assertEqual(majority.arima_count, 2)
        self.assertAlmostEqual(majority.average_confidence, 0.7666666667)
        self.assertEqual(tie.overall_model, "Tie")
        self.assertIn("evenly split", tie.explanation)


if __name__ == "__main__":
    unittest.main()

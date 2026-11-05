"""Exercise the Euclidean notebook branch without expensive neural training."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "ecoli_iML1515_B_model_testing.ipynb"


def summary(value):
    return pd.DataFrame({"avg": [value] * 5, "std": [0.1] * 5},
                        index=["R2", "Pearson_r2", "MAE", "RMSE", "NE"])


class EuclideanNotebookTest(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.source = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        original = pd.DataFrame({"experiment": ["a", "b"],
                                 "flux_a": [1.0, 2.0], "flux_b": [3.0, 4.0]})
        self.fitted = original.copy()
        self.fitted[["flux_a", "flux_b"]] += 5
        self.fitted.to_csv(Path(self.directory.name) / "fluxomics_iML1515_euclidean_fit.csv", index=False)
        self.metrics = pd.DataFrame({"Pearson_r2": [0.8, 0.9]})
        self.loaded = (np.full((2, 1), 5.0), np.full((2, 2), 6.0),
                       np.full((2, 5), 7.0), ["feature"], [], [], ["context"],
                       [], [], ["target"], ["a", "b"])
        self.baseline_calls = []
        self.restore_calls = []
        self.seed_calls = []
        self.fail_training = False
        self.old_best = object()
        self.ns = {
            "np": np, "pd": pd, "Path": Path, "os": os,
            "display": lambda value: None,
            "MINN_DATA_DIR": self.directory.name,
            "MINN_FLUXOMICS_FILE_OPTIONS": {"minn_fitted": "original.csv"},
            "MINN_FLUXOMICS_FILE_MODE": "minn_fitted",
            "MINN_FLUXOMICS_FILE": "original.csv",
            "MINN_GLC_O2_CONTEXT_MODE": "predicted",
            "MINN_SELECTED_FLUXTRANSFORMER_PFBA_RESULT_KEY": "predicted_co2_etoh_ac_cap",
            "MINN_LOO_METRIC_PLOT_SAVE_PATH": "original.png",
            "flux_df": original,
            "outputs": ["token"],
            "minn_feature_names": self.loaded[3],
            "minn_mapped_sources": self.loaded[6],
            "minn_target_mapped_sources": self.loaded[9],
            "minn_experiments": self.loaded[10],
            "minn_cv_seed": 12345,
            "X_minn_np": np.zeros((2, 1)),
            "y_minn_np": np.zeros((2, 2)),
            "y_minn_context_np": np.zeros((2, 5)),
            "best_res": self.old_best,
            "pfba_baseline_pred_df": original.copy(),
            "pfba_baseline_summary_df": summary(0.1),
            "pfba_baseline_metrics_per_sample_df": self.metrics.copy(),
            "IML1515_PFBA_RESULT_KEY": "weighted",
            "IML1515_FITTED_FILE": "weighted.csv",
            "iml1515_fitted_flux_df": original.copy(),
            "iml1515_pfba_baseline_summary_df": summary(0.2),
            "iml1515_pfba_baseline_metrics_per_sample_df": self.metrics.copy(),
            "MINN_CONTEXT_MODE_RESULTS": {},
        }
        self.ns["MINN_FLUXTRANSFORMER_PFBA_RESULTS"] = {
            "measured_co2_etoh_ac_cap": self.result(0.11, "original.csv", original),
            "predicted_co2_etoh_ac_cap": self.result(0.12, "original.csv", original),
            "weighted": self.result(0.21, "weighted.csv", original),
        }
        self.original_state = {name: self.ns[name] for name in (
            "MINN_FLUXOMICS_FILE_MODE", "MINN_FLUXOMICS_FILE", "X_minn_np",
            "y_minn_np", "y_minn_context_np", "pfba_baseline_pred_df",
            "pfba_baseline_summary_df", "MINN_LOO_METRIC_PLOT_SAVE_PATH",
        )}

        def baseline(frame):
            pd.testing.assert_frame_equal(frame, self.fitted)
            self.baseline_calls.append(frame.copy())
            return frame.copy(), self.metrics.copy(), summary(0.3), []

        def load(data_dir, outputs):
            self.assertEqual(self.ns["MINN_FLUXOMICS_FILE"], "fluxomics_iML1515_euclidean_fit.csv")
            return self.loaded

        def train(mode, *, result_key, make_canonical):
            self.assertEqual(mode, "measured")
            self.assertFalse(make_canonical)
            self.assertIs(self.ns["X_minn_np"], self.loaded[0])
            self.assertIs(self.ns["y_minn_np"], self.loaded[1])
            self.assertIs(self.ns["pfba_baseline_summary_df"],
                          self.ns["euclidean_pfba_baseline_summary_df"])
            self.ns.update(best_res="new", rmse_oof=-1, MINN_GLC_O2_CONTEXT_MODE="measured")
            if self.fail_training:
                raise RuntimeError("simulated training failure")
            result = {"mode": mode}
            self.ns["MINN_CONTEXT_MODE_RESULTS"][result_key] = result
            return result

        def pfba(mode, *, extra_constraint_mode, result_key, context_result_key, make_canonical):
            self.assertEqual(extra_constraint_mode, "co2_etoh_ac_cap")
            self.assertFalse(make_canonical)
            self.assertIn(context_result_key, self.ns["MINN_CONTEXT_MODE_RESULTS"])
            self.assertEqual(self.ns["MINN_FLUXOMICS_FILE_MODE"], "iml1515_euclidean")
            result = self.result(0.31, self.ns["MINN_FLUXOMICS_FILE"], self.fitted)
            self.ns["MINN_FLUXTRANSFORMER_PFBA_RESULTS"][result_key] = result
            self.ns["MINN_SELECTED_FLUXTRANSFORMER_PFBA_RESULT_KEY"] = result_key
            return result

        def configure(mode):
            self.ns["MINN_GLC_O2_CONTEXT_MODE"] = mode

        def restore(result):
            self.restore_calls.append(result)
            self.ns.update(best_res=self.old_best, MINN_GLC_O2_CONTEXT_MODE="predicted",
                           MINN_SELECTED_FLUXTRANSFORMER_PFBA_RESULT_KEY="predicted_co2_etoh_ac_cap")

        self.ns.update(evaluate_pfba_baseline=baseline, load_minn_omics_for_reservoir=load,
                       run_minn_context_mode_experiment=train,
                       run_fluxtransformer_to_pfba_context_mode=pfba,
                       configure_minn_glc_o2_context_mode=configure,
                       _apply_minn_fluxtransformer_pfba_result=restore,
                       set_seed=self.seed_calls.append)

    def result(self, value, filename, truth):
        return {"fluxomics_file": filename, "minn_pfba_summary_df": summary(value),
                "minn_pfba_metrics_per_sample_df": self.metrics.copy(),
                "flux_df_eval": truth.copy()}

    def execute(self, cell_id):
        exec(compile(self.source[cell_id], str(NOTEBOOK), "exec"), self.ns)

    def test_paired_euclidean_results_and_seven_row_final_table(self):
        weighted = self.ns["MINN_FLUXTRANSFORMER_PFBA_RESULTS"]["weighted"]
        self.execute("minn_euclidean_baseline")
        self.execute("minn_euclidean_reservoir")
        for name, value in self.original_state.items():
            self.assertIs(self.ns[name], value, name)
        self.assertIs(self.ns["MINN_FLUXTRANSFORMER_PFBA_RESULTS"]["weighted"], weighted)
        self.assertEqual(self.seed_calls, [12345])
        self.assertEqual(len(self.restore_calls), 1)
        self.execute("minnfit60")
        table = self.ns["MINN_FINAL_PFBA_AVERAGE_COMPARISON_DF"]
        self.assertEqual(len(table), 7)
        self.assertEqual(table["Fitted file"].tolist()[-2:], ["iML1515 Euclidean fit"] * 2)
        self.assertEqual(table["Pearson r2"].tolist()[-2:],
                         ["0.300000 +/- 0.100000", "0.310000 +/- 0.100000"])
        self.assertEqual(table["Successful"].tolist(), ["2/2"] * 7)

    def test_failure_restores_state_without_original_context_results(self):
        self.ns.pop("MINN_SELECTED_FLUXTRANSFORMER_PFBA_RESULT_KEY")
        self.fail_training = True
        self.execute("minn_euclidean_baseline")
        with self.assertRaisesRegex(RuntimeError, "simulated training failure"):
            self.execute("minn_euclidean_reservoir")
        for name, value in self.original_state.items():
            self.assertIs(self.ns[name], value, name)
        self.assertIs(self.ns["best_res"], self.old_best)
        self.assertNotIn("rmse_oof", self.ns)
        self.assertNotIn("MINN_SELECTED_FLUXTRANSFORMER_PFBA_RESULT_KEY", self.ns)
        self.assertEqual(self.ns["MINN_GLC_O2_CONTEXT_MODE"], "predicted")

    def test_final_table_rejects_mixed_fitted_targets(self):
        self.execute("minn_euclidean_baseline")
        self.execute("minn_euclidean_reservoir")
        result = self.ns["MINN_FLUXTRANSFORMER_PFBA_RESULTS"][self.ns["EUCLIDEAN_PFBA_RESULT_KEY"]]
        result["flux_df_eval"].loc[0, "flux_a"] += 1
        with self.assertRaisesRegex(RuntimeError, "targets differ"):
            self.execute("minnfit60")


if __name__ == "__main__":
    unittest.main()

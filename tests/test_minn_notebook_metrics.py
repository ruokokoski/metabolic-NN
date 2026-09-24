"""Check the standalone MINN notebook's Table 4 metric definition."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


class MinnNotebookMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "ecoli_iML1515_MINN_model_testing.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        cls.functions = {}
        for cell in cells:
            source = "".join(cell["source"])
            if not any(f"def {name}(" in source for name in
                       ("_metric_row", "metric_row_fn", "_refresh_pfba_r2")):
                continue
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.FunctionDef) and node.name in (
                    "_metric_row", "metric_row_fn", "_refresh_pfba_r2"
                ):
                    namespace = dict(np=np, pd=pd, r2_score=r2_score,
                                     mapped_metrics=[(c, c, 1) for c in ("a", "b", "c")])
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                    cls.functions[node.name] = namespace[node.name]

    def test_regression_not_correlation(self):
        truth = np.array([1., 2., 3.])
        for name in ("_metric_row", "metric_row_fn"):
            for prediction in (truth, 2 * truth + 5, np.full(3, truth.mean())):
                self.assertEqual(self.functions[name](truth, prediction)[0],
                                 r2_score(truth, prediction, force_finite=False))
            self.assertLess(self.functions[name](truth, 2 * truth + 5)[0], 0)

    def test_cached_rescoring_aligns_experiments_and_preserves_other_metrics(self):
        truth = pd.DataFrame({"experiment": ["x", "y"], "a": [1., 2.],
                              "b": [2., 4.], "c": [3., 6.]})
        predicted = truth.iloc[::-1].copy()
        predicted.loc[:, ["a", "b", "c"]] += 1
        metrics = pd.DataFrame({"experiment": ["x", "y"], "R2": [1., 1.], "MAE": [1., 1.]})
        summary = pd.DataFrame({"avg": [1., 1.], "std": [0., 0.]}, index=["R2", "MAE"])
        self.functions["_refresh_pfba_r2"](predicted, truth, metrics, summary)
        np.testing.assert_allclose(metrics.R2, [-0.5, 0.625])
        self.assertEqual(summary.loc["R2", "avg"], 0.0625)
        self.assertEqual(summary.loc["R2", "std"], 0.5625)
        np.testing.assert_array_equal(metrics.MAE, [1., 1.])
        np.testing.assert_array_equal(summary.loc["MAE"], [1., 0.])


if __name__ == "__main__":
    unittest.main()

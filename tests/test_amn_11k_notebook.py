"""Check that the AMN_11k notebook preserves A's analysis protocol."""

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def code_tree(cell):
    source = "".join(line for line in cell["source"] if not line.lstrip().startswith("%"))
    return ast.parse(source)


def definitions(notebook):
    result = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in code_tree(cell).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    result.append(ast.dump(node, include_attributes=False))
    return result


class AMN11kNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads((ROOT / "ecoli_iML1515_A_model_testing.ipynb").read_text())
        cls.amn = json.loads((ROOT / "ecoli_iML1515_AMN_model_testing.ipynb").read_text())

    def test_analysis_definitions_and_cell_order_match_a(self):
        self.assertEqual(len(self.original["cells"]), len(self.amn["cells"]))
        self.assertEqual(definitions(self.original), definitions(self.amn))
        configuration_cells = {0, 13, 15, 36, 48, 49, 52, 56}
        for index, (original, current) in enumerate(zip(self.original["cells"], self.amn["cells"])):
            self.assertEqual(original["cell_type"], current["cell_type"])
            if index not in configuration_cells:
                self.assertEqual(original["source"], current["source"], f"Analysis drift in cell {index}")

    def test_new_model_data_and_artifacts_are_separate(self):
        cell_ids = [cell["id"] for cell in self.amn["cells"]]
        self.assertEqual(len(set(cell_ids)), len(cell_ids))
        assignments = {}
        for cell in self.amn["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            for node in code_tree(cell).body:
                if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assignments[target.id] = node.value
        self.assertEqual(ast.literal_eval(assignments["model_name"]), "AMN_11k_d256_h8_l4_ff1024")
        data_path = assignments["DATA_PATH"]
        if isinstance(data_path, ast.JoinedStr):
            data_path = "".join(ast.literal_eval(value) for value in data_path.values)
        else:
            data_path = ast.literal_eval(data_path)
        self.assertEqual(data_path, "./data/iML1515_AMN_11k_test_data_11000_samples.csv")
        self.assertEqual(ast.literal_eval(assignments["RANDOM_AMN_TEST_PATH"]), data_path)
        self.assertEqual(ast.literal_eval(assignments["N_EVAL_SAMPLES"]), 11000)
        self.assertEqual(ast.literal_eval(assignments["RANDOM_AMN_CONTEXTS"]), 10000)
        for index in (48, 49, 56):
            self.assertIn('Path(f"./insights/thesis/{model_name}")', "".join(self.amn["cells"][index]["source"]))
        validation = "".join(self.amn["cells"][15]["source"])
        self.assertIn('list(outputs) == list(data_info["output_cols"])', validation)
        self.assertIn('os.path.realpath(DATA_PATH) != os.path.realpath(data_info["dataset"])', validation)


if __name__ == "__main__":
    unittest.main()

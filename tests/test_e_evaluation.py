"""Regression checks for revised E media and full-vocabulary shift inputs."""

import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

import generate_ecoli_iML1515_E_data_old as archived
import iml1515_evaluation as ev


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("legacy", [False, True])
def test_e_minn_basal_contract_without_cobalamin(legacy):
    from cobra.io import read_sbml_model
    outputs = [reaction.id + "_flux" for reaction in read_sbml_model(str(ROOT / "models/iML1515.xml")).reactions]
    data = ev.load_minn(ROOT / "MINN_data", outputs, model_family="E", legacy_protocol=legacy)
    assert set(data["fixed"]) == set(ev.COMMON_BASE_EXCHANGES)
    assert set(data["fixed"].values()) == {10.0}
    assert "EX_cbl1_e" not in data["fixed"]


def test_archived_e_checkpoint_is_rejected(tmp_path):
    inputs = archived.build_input_columns()
    path = tmp_path / "old_checkpoint.pth"
    torch.save(dict(model_state_dict={}, config=dict(d_model=4, n_heads=2, n_layers=1,
        d_ff=8, dropout=0, vocab_size=len(inputs)),
        data_info=dict(input_cols=inputs, output_cols=[name + "_flux" for name in inputs])), path)
    with pytest.raises(ValueError, match="Checkpoint input order does not match model E generator"):
        ev.load_reservoir(path, "cpu", model_family="E")


def test_e_notebook_schema_and_source_medium_injection(tmp_path):
    notebook = nbformat.read(ROOT / "ecoli_iML1515_E_model_testing.ipynb", as_version=4)
    nbformat.validate(notebook)
    function = None
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        compile(tree, f"E_cell_{i}", "exec")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "compute_overall_model_metrics":
                function = node
    assert function is not None

    class RecordingReservoir(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.inputs = []

        def forward(self, context, output_subset=None):
            self.inputs.append(context.clone())
            return torch.zeros_like(context), torch.arange(context.shape[1])

    reservoir = RecordingReservoir()
    path = tmp_path / "B_medium.csv"
    pd.DataFrame({"EX_pi_e": [50.0], "EX_cbl1_e": [50.0],
                  "EX_pi_e_flux": [0.0], "EX_cbl1_e_flux": [0.0]}).to_csv(path, index=False)
    namespace = dict(Path=Path, pd=pd, np=np, torch=torch, ev=ev, reservoir=reservoir,
        training_info={}, input_names=["EX_pi_e"], output_names=["EX_pi_e_flux", "EX_cbl1_e_flux"])
    exec(compile(ast.Module(body=[function], type_ignores=[]), "E_metrics", "exec"), namespace)
    result = namespace["compute_overall_model_metrics"](
        path, torch.device("cpu"), 1, source_input_names=["EX_pi_e", "EX_cbl1_e"])
    assert result["Rows"] == 1
    assert result["RMSE"] == 0
    np.testing.assert_array_equal(reservoir.inputs[0].numpy().reshape(-1), [50, 50])

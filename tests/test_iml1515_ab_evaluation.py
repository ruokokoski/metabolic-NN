"""Small CPU checks for the combined evaluation's scientific contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import iml1515_ab_evaluation as ev
from flux_transformer import FluxTransformer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup():
    import pandas as pd
    source = pd.read_csv(ROOT / "MINN_data" / ev.FLUX_FILES["minn_fitted"], nrows=0)
    tokens = [f"{name}_flux" for name in ev.build_input_columns()]
    for name in source.columns[1:]:
        if name.startswith("R_BIOMASS"):
            token = f"{ev.OBJECTIVE}_flux"
        else:
            token = name.removeprefix("R_").removesuffix("_fwd").removesuffix("_rev") + "_flux"
        if token not in tokens:
            tokens.append(token)
    ev.seed_all(10)
    model = FluxTransformer(len(tokens), list(range(41)), d_model=4, n_heads=2, n_layers=1, d_ff=8, dropout=0)
    model.eval().requires_grad_(False)
    return model, tokens, ev.load_amn(ROOT / "AMN_data"), ev.load_minn(ROOT / "MINN_data", tokens)


def settings():
    return dict(hidden=8, epochs=1, patience=1, min_delta=0, batch_size=2, delta=1.0,
                clip=1.0, amp=False, warmup=1)


def test_real_data_and_task_medium_contracts(setup):
    model, tokens, amn, minn = setup
    assert amn["X"].shape == (110, 11)
    assert minn["X"].shape == (29, 141)
    assert minn["y"].shape == (29, 42)
    assert len(minn["mapping"]) == 47
    a = ev.FrontReservoir(model, tokens, amn, "amn")
    b = ev.FrontReservoir(model, tokens, minn, "minn")
    ac, av = a.medium(torch.from_numpy(amn["X"][:2]), torch.from_numpy(amn["observed"][:2]))
    bc, bv = b.medium(torch.from_numpy(minn["X"][:2]), torch.from_numpy(minn["observed"][:2]))
    assert ac[0, tokens.index("EX_pi_e_flux")] == 10
    assert ac[0, tokens.index("EX_cbl1_e_flux")] == 0
    assert bc[0, tokens.index("EX_pi_e_flux")] == 50
    assert bc[0, tokens.index("EX_cbl1_e_flux")] == 50
    assert bc[0, tokens.index("EX_glyc_e_flux")] == 0
    np.testing.assert_allclose(bv[:, :2].detach(), minn["observed"][:2])
    assert torch.all(av[torch.from_numpy(amn["X"][:2]) == 0] == 0)
    assert a.front[0].weight.data_ptr() != b.front[0].weight.data_ptr()


def test_gradient_reaches_both_minn_fronts_but_reservoir_unchanged(setup):
    reservoir, tokens, _, data = setup
    before = {k: v.clone() for k, v in reservoir.state_dict().items()}
    for mode, count in [("measured", 3), ("predicted", 5)]:
        model = ev.FrontReservoir(reservoir, tokens, data, "minn", mode, hidden=8)
        model.train()
        assert not reservoir.training
        x = torch.from_numpy(data["X"][:2] * 0.01)
        prediction, context = model(x, torch.from_numpy(data["observed"][:2]))
        torch.nn.functional.huber_loss(prediction, torch.from_numpy(data["y"][:2])).backward()
        assert model.front[-1].out_features == count and context.shape == (2, 5)
        assert sum(p.grad.abs().sum() for p in model.front.parameters()) > 0
        assert all(p.grad is None for p in reservoir.parameters())
        torch.optim.SGD(model.front.parameters(), lr=0.01).step()
    assert all(torch.equal(before[k], v) for k, v in reservoir.state_dict().items())


def test_scaler_uses_only_training_rows_and_rejects_overlap(setup):
    reservoir, tokens, _, data = setup
    params = dict(drop_rate=0, learning_rate=0.001, weight_decay=0)
    model, scaler, epoch, _, history = ev.fit_front(reservoir, tokens, data, "minn", "measured",
        [0, 1, 2], settings(), params, 10, validation_ids=[3])
    np.testing.assert_array_equal(scaler.data_min_, data["X"][:3].min(axis=0))
    assert epoch == 1 and len(history) == 1
    with pytest.raises(ValueError, match="overlap"):
        ev.fit_front(reservoir, tokens, data, "minn", "measured", [0, 1], settings(), params, 10, validation_ids=[1])


def test_checkpoint_schema_rejects_reordering(setup, tmp_path):
    model, tokens, _, _ = setup
    checkpoint = dict(model_state_dict=model.state_dict(), config=dict(d_model=4, n_heads=2,
        n_layers=1, d_ff=8, dropout=0, vocab_size=len(tokens)),
        data_info=dict(input_cols=ev.build_input_columns(), output_cols=tokens))
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)
    loaded, *_ = ev.load_reservoir(path, "cpu")
    assert not any(p.requires_grad for p in loaded.parameters())
    checkpoint["data_info"]["input_cols"] = list(reversed(ev.build_input_columns()))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="input order"):
        ev.load_reservoir(path, "cpu")


def test_metric_definitions():
    result = ev.metrics([1, 2, 3], [2, 4, 6])
    assert result["Pearson_r2"] == pytest.approx(1)
    assert result["R2"] < 0
    assert np.isnan(ev.metrics([0, 0], [1, 1])["NE"])
    assert np.isnan(ev.metrics([1, 1], [1, 1])["R2"])


def test_amn_summary_scores_mean_predictions_like_model_c():
    import pandas as pd
    oof = pd.DataFrame({"row": [0, 1, 0, 1], "repeat": [10, 10, 11, 11],
                        "truth": [1., 2., 1., 2.], "prediction": [0., 1., 2., 3.]})
    means, scores, spread = ev.summarize_amn_oof(oof, 2, [10, 11])
    assert scores["R2"] == 1 and scores["MAE"] == scores["RMSE"] == 0
    assert spread["R2"] == 0
    np.testing.assert_allclose(means.prediction_std, np.sqrt(2))
    # Averaging the repeat R2 scores would give -3, not 1.
    assert ev.metrics([1, 2], [0, 1])["R2"] == -3
    with pytest.raises(ValueError, match="Incomplete"):
        ev.summarize_amn_oof(oof.iloc[:-1], 2, [10, 11])


def test_amp_overflow_retries_once_with_same_seed_in_fp32(monkeypatch):
    from types import SimpleNamespace
    reservoir = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))]))
    calls = []

    def fake_fit(*args):
        calls.append(args)
        if args[6]["amp"]:
            raise FloatingPointError("overflow")
        return "finite fit"

    monkeypatch.setattr(ev, "_fit_front", fake_fit)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    config = settings() | {"amp": True}
    assert ev.fit_front(reservoir, [], {}, "minn", "measured", [0], config, {}, 42) == "finite fit"
    assert len(calls) == 2 and calls[0][8] == calls[1][8] == 42
    assert calls[1][6]["amp"] is False and config["amp"] is True


def test_nonfinite_gradients_cannot_update_front_weights(setup, monkeypatch):
    reservoir, tokens, _, data = setup
    original = ev.FrontReservoir
    captured = []

    def overflowing_front(*args, **kwargs):
        model = original(*args, **kwargs)
        captured.append((model, {k: v.clone() for k, v in model.front.state_dict().items()}))
        model.front[0].weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return model

    monkeypatch.setattr(ev, "FrontReservoir", overflowing_front)
    with pytest.raises(FloatingPointError, match="gradients"):
        ev.fit_front(reservoir, tokens, data, "minn", "measured", [0, 1], settings(),
                     dict(drop_rate=0, learning_rate=0.001, weight_decay=0), 10)
    model, before = captured[0]
    assert all(torch.equal(before[k], v) for k, v in model.front.state_dict().items())


def test_global_minn_hpo_once_and_loo_early_stopping(setup, tmp_path):
    reservoir, tokens, _, original = setup
    data = dict(original)
    for key in ("X", "y", "observed", "ids"):
        data[key] = data[key][:4]
    config = settings() | dict(seed=12345, inner_folds=2, trials=1, std_penalty=0.25,
        drop_rates=[0.0], lr_range=[0.001, 0.002], wd_range=[1e-8, 1e-6])
    result = ev.run_minn(reservoir, tokens, data, "predicted", config, tmp_path)
    assert result["prediction"].shape == (4, 42)
    assert np.isfinite(result["context"]).all()
    for fold in result["folds"]:
        assert not set(fold["train"]) & set(fold["test"])
        assert fold["hpo_scope"] == "full_dataset"
        assert fold["early_stopping_scope"] == "outer_test"
        for inner in fold["hpo_splits"]:
            assert not set(inner["train"]) & set(inner["validation"])
            assert set(inner["train"] + inner["validation"]) == set(range(4))
    assert (tmp_path / "oof_context.csv").is_file()
    import pandas as pd
    assert len(pd.read_csv(tmp_path / "global_hpo_trials.csv")) == 1
    assert not list(tmp_path.glob("fold_*_trials.csv"))
    assert all(fold["params"] == result["folds"][0]["params"] for fold in result["folds"])


def test_amn_cv_smoke(setup, tmp_path):
    reservoir, tokens, original, _ = setup
    data = dict(original)
    # Retain all 110 real media; two folds and one epoch keep this test small.
    config = settings() | dict(folds=2, split_seeds=[10], train_seed=10, inner_fraction=0.2,
        params=dict(drop_rate=0, learning_rate=0.001, weight_decay=0.001))
    result = ev.run_amn(reservoir, tokens, data, config, tmp_path)
    assert len(result["oof"]) == 110
    for path in tmp_path.glob("*_fold.json"):
        fold = json.loads(path.read_text())
        assert not set(fold["test"]) & set(fold["inner_train"] + fold["inner_validation"])


def test_pfba_uses_observed_uptake_in_both_context_modes(setup, tmp_path):
    _, _, _, original = setup
    data = dict(original)
    for key in ("ids", "observed", "context_truth"):
        data[key] = data[key][:2]
    data["flux"] = data["flux"].iloc[:2]
    context = data["context_truth"].copy()
    context[:, 2:] = [100, 100, 100]
    a = ev.run_pfba(ROOT / "models/iML1515.xml", data, context)
    context[:, :2] = 1000  # These internal predictions must never become pFBA uptake caps.
    b = ev.run_pfba(ROOT / "models/iML1515.xml", data, context)
    baseline = ev.run_pfba(ROOT / "models/iML1515.xml", data)
    assert a["success"].all() and b["success"].all() and baseline["success"].all()
    np.testing.assert_allclose(a["prediction"], b["prediction"])
    table = ev.compare_pfba({"baseline": baseline, "measured": a, "predicted": b}, data, tmp_path)
    assert set(table.method) == {"baseline", "measured", "predicted"}
    assert (table.successes == 2).all()


def test_notebook_valid_and_no_forbidden_benchmark():
    import nbformat
    notebook = nbformat.read(ROOT / "ecoli_iML1515_AB_union_model_testing.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert "tabpfn" not in json.dumps(notebook).lower()
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell_{i}", "exec")
            assert cell.execution_count is None and not cell.outputs


def test_notebook_preflight_allows_unknown_provenance_and_optional_files(tmp_path):
    notebook = json.loads((ROOT / "ecoli_iML1515_AB_union_model_testing.ipynb").read_text())
    preflight = next("".join(c["source"]) for c in notebook["cells"]
                     if "required = {\"CHECKPOINT\"" in "".join(c["source"]))
    placeholder = tmp_path / "required_artifact"
    placeholder.touch()
    scope = dict(Path=Path, CHECKPOINT=placeholder, XML_PATH=placeholder,
        TRAINING_LOG=None, METADATA_JSON=None, BIOMASS_TEST_PATH=None,
        RUN_AMN=False, RUN_MINN=False, RUN_SIMULATED=False,
        TRAINING_PROVENANCE={"generation_command": "", "training_command": "", "seed": 42})
    exec(preflight.split("reservoir, input_names")[0], scope)
    assert set(scope["required"]) == {"CHECKPOINT", "XML"}


def test_simulated_regime_bounds_and_objective_checks(tmp_path):
    from types import SimpleNamespace
    import pandas as pd
    from cobra.flux_analysis import pfba
    import generate_ecoli_iML1515_AB_union_data as generator

    gem, defaults, reactions, _ = generator.load_generation_model(
        ROOT / "models", ev.OBJECTIVE, 120)
    inputs, outputs = generator.build_input_columns(), [f"{r}_flux" for r in reactions]
    args = SimpleNamespace(a_base_rate=10, a_oxygen_rate_min=1, a_oxygen_rate_max=10,
        a_max_carbon_sources=4, a_carbon_rate_min=0.05, a_carbon_rate_max=2.2,
        a_fixed_carbon_rate=2.2, a_amino_rate=2.2, b_base_rate=50)
    b_caps = dict(zip(ev.MINN_CONTEXT_EXCHANGES, [(1, 15), (1, 20), (0, 15), (0, 1), (0, 3)]))
    truth, specs = [], {}
    for domain in ("A", "B"):
        row = dict.fromkeys(inputs, 0.0)
        if domain == "A":
            generator.apply_a_regime(gem, row, np.random.default_rng(9), defaults, args)
        else:
            generator.apply_b_regime(gem, row, np.random.default_rng(9), defaults, b_caps, args)
        solution = pfba(gem, fraction_of_optimum=0.999)
        values = solution.fluxes[reactions].to_numpy()
        truth.append(values)
        row.update(dict(zip(outputs, values)))
        path = tmp_path / f"{domain}.csv"
        pd.DataFrame([row]).to_csv(path, index=False)
        specs[domain] = dict(path=path, seed=9, rows=1, command="test fixture from actual regime generator")

    class KnownFluxes(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.register_buffer("truth", torch.tensor(np.array(truth), dtype=torch.float32))

        def forward(self, c, output_subset=None):
            assert output_subset is None
            regime = (c[:, outputs.index("EX_glc__D_e_flux"), 0] > 0).long()
            return self.truth[regime].unsqueeze(-1), torch.arange(len(outputs))

    result = ev.simulated_fidelity(KnownFluxes(), inputs, outputs, specs,
        ROOT / "models/iML1515.xml", tmp_path / "metrics", objective_samples=1)
    assert set(result.domain) == {"A", "B", "balanced_union"}
    assert result.RMSE.max() < 1e-5
    physical = pd.read_csv(tmp_path / "metrics/physical_diagnostics.csv")
    assert physical.bound_violation_max.max() < 1e-5
    assert physical.mass_balance_max.max() < 1e-4
    assert (physical.objective_status == "optimal").all()
    assert physical.prediction_minus_pfba_objective.abs().max() < 1e-5


@pytest.mark.parametrize("family,exclude", [("C", True), ("C", False), ("D", False), ("E", False)])
def test_shared_model_contracts_and_media(tmp_path, family, exclude):
    inputs = ev.input_contract(family, exclude_cbl1=exclude)
    tokens = [f"{name}_flux" for name in inputs]
    model = FluxTransformer(len(tokens), list(range(len(inputs))), d_model=4,
                            n_heads=2, n_layers=1, d_ff=8, dropout=0)
    path = tmp_path / "checkpoint.pth"
    torch.save(dict(model_state_dict=model.state_dict(), config=dict(d_model=4,
        n_heads=2, n_layers=1, d_ff=8, dropout=0, vocab_size=len(tokens)),
        data_info=dict(input_cols=inputs, output_cols=tokens)), path)
    _, loaded_inputs, *_ = ev.load_reservoir(path, "cpu", model_family=family)
    assert loaded_inputs == inputs
    amn = ev.load_amn(ROOT / "AMN_data", input_names=inputs, model_family=family)
    assert amn["fixed"]["EX_pi_e"] == (10 if family == "D" else 50)
    if exclude:
        assert "EX_cbl1_e" not in amn["fixed"]
    else:
        assert amn["fixed"]["EX_cbl1_e"] == (0 if family == "D" else 50)


def test_legacy_amn_matches_original_notebook_training(setup):
    """Compare restored training directly with the original notebook functions."""
    import ast
    from copy import deepcopy
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.checkpoint import checkpoint

    reservoir, tokens, _, _ = setup
    inputs = ev.build_input_columns()
    data = ev.load_amn(ROOT / "AMN_data", inputs, "C", feature_order="checkpoint")
    variable = data["features"]
    assert variable == [name for name in inputs if name in ev.AMN_CARBON_EXCHANGES + ["EX_o2_e"]]
    fixed = [name for name in inputs if name not in variable]
    ns = dict(torch=torch, nn=torch.nn, np=np, deepcopy=deepcopy,
              DataLoader=DataLoader, TensorDataset=TensorDataset, checkpoint=checkpoint,
              set_seed=ev.seed_all, device=torch.device("cpu"), model=reservoir,
              inputs=inputs, outputs=tokens, variable_input_cols=variable,
              hidden_dim=8, dropout=0.0, num_epochs=3, patience=15,
              out_scale_vec=np.array([10.0 if c == "EX_o2_e" else 2.2 for c in variable], np.float32),
              var_pos=torch.tensor([inputs.index(c) for c in variable]),
              fixed_pos=torch.tensor([inputs.index(c) for c in fixed]),
              fixed_values=torch.tensor([data["fixed"][c] for c in fixed]),
              token_idx=reservoir.input_token_indices,
              biomass_idx=tokens.index(data["targets"][0]))
    original = json.loads((ROOT / "ecoli_iML1515_AMN_MINN_model_testing_trial.ipynb").read_text(encoding="utf-8"))
    wanted = {"PriorDenseNetwork", "compose_full_medium", "medium_to_reservoir_input",
              "_model_only", "_predict_growth_with_prior", "train_prior_one_seed"}
    for cell in original["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not any(f"def {name}(" in source or f"class {name}(" in source for name in wanted):
            continue
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
        exec(compile(tree, "original_notebook", "exec"), ns)
    x, y = torch.from_numpy(data["X"]), torch.from_numpy(data["y"][:, 0])
    reference = ns["train_prior_one_seed"](10, x[:8], y[:8], x[8:13], y[8:13], 1, 2, "parity", True)
    config = dict(settings(), epochs=3, patience=15, batch_size=1, validation_batch_size=2,
                  delta=0.03, clip=None, validation_protocol="legacy_outer_early_stopping")
    params = dict(drop_rate=0, learning_rate=0.001, weight_decay=0.001)
    actual, _, epoch, loss, history = ev.fit_front(reservoir, tokens, data, "amn", "measured",
        np.arange(8), config, params, 10, validation_ids=np.arange(8, 13))
    assert epoch == reference["best_epoch"]
    np.testing.assert_allclose([r["validation_loss"] for r in history], reference["val_losses"], rtol=1e-6)
    for key, value in actual.front.state_dict().items():
        torch.testing.assert_close(value, reference["model"].net.state_dict()[key], rtol=1e-6, atol=1e-7)
    prediction, _ = ev.predict_front(actual, data["X"][8:13], data["observed"][8:13], 2)
    np.testing.assert_allclose(prediction[:, 0], reference["y_val_pred"], rtol=1e-6)


def test_legacy_amn_scores_selected_fold_model(setup, tmp_path, monkeypatch):
    import iml1515_evaluation as shared
    reservoir, tokens, data, _ = setup
    calls = []
    def fit(*args, **kwargs):
        train = np.asarray(args[5])
        validation = np.asarray(kwargs["validation_ids"])
        assert not np.intersect1d(train, validation).size
        assert "epochs" not in kwargs
        calls.append((train, validation))
        return shared.FrontReservoir(reservoir, tokens, data, "amn", hidden=8), None, 1, 0.0, []
    monkeypatch.setattr(shared, "fit_front", fit)
    config = dict(settings(), folds=10, split_seeds=[10], train_seed=10,
                  validation_protocol="legacy_outer_early_stopping", params={})
    result = shared.run_amn(reservoir, tokens, data, config, tmp_path)
    assert len(calls) == 10 and len(result["oof"]) == 110
    for fold, (train, validation) in enumerate(calls, 1):
        record = json.loads((tmp_path / f"repeat_10_fold_{fold}_fold.json").read_text())
        assert len(train) == 99 and len(validation) == 11
        assert record["test"] == record["selection_validation"] == validation.tolist()
        assert record["inner_validation"] == record["refit_history"] == []


@pytest.mark.parametrize("mode", ["measured", "predicted"])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_legacy_minn_matches_original_fit_and_rng(setup, mode, device_name):
    import ast
    from sklearn.preprocessing import MinMaxScaler
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    reservoir, tokens, _, data = setup
    device = torch.device(device_name)
    reservoir.to(device)
    data["legacy_minn_protocol"] = True
    original = json.loads((ROOT / "ecoli_iML1515_MINN_AMN_model_testing_trial.ipynb").read_text(encoding="utf-8"))
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, np=np, outputs=tokens,
              MinMaxScaler=MinMaxScaler, minn_cv_early_stopping_patience=2,
              minn_cv_early_stopping_min_delta=1e-5, MINN_GRAD_CLIP_MAX_NORM=1.0,
              MINN_LR_WARMUP_EPOCHS=2, MINN_LR_COSINE_MIN_FACTOR=0.05,
              minn_target_indices_t=torch.tensor([tokens.index(t) for t in data["targets"]]),
              minn_context_signs_t=torch.tensor([-1., -1., 1., 1., 1.]),
              minn_measured_context_positions_t=torch.tensor([0, 1] if mode=="measured" else [], dtype=torch.long),
              minn_predicted_context_positions_t=torch.tensor([2, 3, 4] if mode=="measured" else [0, 1, 2, 3, 4]))
    wanted = {"FrozenFluxTransformerWithMLP", "_train_eval_split"}
    for cell in original["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or not any(f"def {s}(" in source or f"class {s}(" in source for s in wanted):
            continue
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
        exec(compile(tree, "original_minn_notebook", "exec"), ns)
    models = []
    def make_model(drop_rate, device_obj):
        context = [s.removeprefix("R_").removesuffix("_rev").removesuffix("_fwd") + "_flux" for s in ev.CONTEXT_SOURCES]
        m = ns["FrozenFluxTransformerWithMLP"](reservoir, data["X"].shape[1],
            context[2:] if mode=="measured" else context, context[:2] if mode=="measured" else [],
            [139, 140] if mode=="measured" else [], {k+"_flux":v for k,v in data["fixed"].items()},
            hidden_dim=8, drop_rate=drop_rate).to(device_obj)
        models.append(m)
        return m
    ns["_make_model"] = make_model
    params = dict(drop_rate=0.2, learning_rate=0.001, weight_decay=0.0001)
    cfg = dict(settings(), legacy_protocol=True, epochs=3, patience=2, min_delta=1e-5,
               delta=1.0, warmup=2, hidden=8)
    context = data["context_truth"].astype(np.float32) * np.array([-1,-1,1,1,1],np.float32)
    ev.seed_all(12345)
    reference = ns["_train_eval_split"](data["X"][:8], data["y"][:8], context[:8],
        data["X"][8:11], data["y"][8:11], context[8:11], params, 3, 2, device)
    expected_rng = torch.get_rng_state().clone()
    expected_cuda_rng = torch.cuda.get_rng_state().clone() if device_name=="cuda" else None
    ev.seed_all(12345)
    actual, _, _, loss, _ = ev.fit_front(reservoir, tokens, data, "minn", mode,
        np.arange(8), cfg, params, 999, validation_ids=np.arange(8,11))
    assert torch.equal(expected_rng, torch.get_rng_state())
    if expected_cuda_rng is not None:
        assert torch.equal(expected_cuda_rng, torch.cuda.get_rng_state())
    np.testing.assert_allclose(loss, reference[0], rtol=1e-6)
    pred, caps = actual.legacy_validation
    np.testing.assert_allclose(pred, reference[1], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(caps, reference[3], rtol=1e-6, atol=1e-7)
    for key,value in actual.front.state_dict().items():
        torch.testing.assert_close(value,models[0].front_mlp.state_dict()[key],rtol=1e-6,atol=1e-7)


def test_legacy_cobalamin_and_pfba_summary(setup):
    reservoir, tokens, _, _ = setup
    inputs = [x for x in ev.build_input_columns() if x != "EX_cbl1_e"]
    current = ev.load_minn(ROOT / "MINN_data", tokens, "minn_fitted", input_names=inputs, model_family="C")
    legacy = ev.load_minn(ROOT / "MINN_data", tokens, "minn_fitted", input_names=inputs, model_family="C", legacy_protocol=True)
    assert "EX_cbl1_e" not in current["fixed"] and legacy["fixed"]["EX_cbl1_e"] == 50
    reservoir.input_token_indices = torch.tensor([i for i in reservoir.input_token_indices if tokens[i] != "EX_cbl1_e_flux"])
    model = ev.FrontReservoir(reservoir, tokens, legacy, "minn", hidden=8)
    medium, _ = model.medium(torch.from_numpy(legacy["X"][:1]), torch.from_numpy(legacy["observed"][:1]))
    assert medium[0,tokens.index("EX_cbl1_e_flux")] == 50
    result = dict(truth=np.array([[1.,2.,3.],[2.,4.,6.]]), prediction=np.array([[2.,4.,6.],[4.,8.,12.]]), success=np.array([True,True]))
    summary = ev.legacy_pfba_summary(result)
    assert summary.loc["R2","avg"] == pytest.approx(1)
    assert summary.loc["MAE","avg"] == pytest.approx(3)
    assert summary.loc["MAE","std"] == pytest.approx(1)


def test_ab_union_matches_corrected_c_notebook_protocol():
    """Every executable cell must match C except explicit model identity strings."""
    ab = json.loads((ROOT / "ecoli_iML1515_AB_union_model_testing.ipynb").read_text(encoding="utf-8"))
    c = json.loads((ROOT / "ecoli_iML1515_C_model_testing.ipynb").read_text(encoding="utf-8"))
    replacements = {
        'MODEL_FAMILY = "AB_union"': 'MODEL_FAMILY = "C"',
        "AB_1M_d256_h8_l4_ff1024": "AMN_MINN_1M_d256_h8_l4_ff1024",
        "iML1515_AB_union_test_data": "iML1515_AMN_MINN_test_data",
        '"AB_union_evaluation"': '"C_evaluation"',
        "generate_ecoli_iML1515_AB_union_data.py": "generate_ecoli_iML1515_AMN_MINN_data.py",
        "ecoli_iML1515_AB_union_model_testing.ipynb": "ecoli_iML1515_C_model_testing.ipynb",
        "A-union-B simulated test CSV": "model C simulated test CSV",
    }
    assert len(ab["cells"]) == len(c["cells"]) == 40
    allowed_markdown_differences = {0, 2, 17, 19, 23}
    for i, (actual, reference) in enumerate(zip(ab["cells"], c["cells"])):
        assert actual["cell_type"] == reference["cell_type"]
        source = "".join(actual["source"])
        if actual["cell_type"] == "code":
            for before, after in replacements.items():
                source = source.replace(before, after)
            assert source == "".join(reference["source"]), f"Protocol drift in cell {i}"
        elif i not in allowed_markdown_differences:
            assert source == "".join(reference["source"]), f"Unreviewed markdown drift in cell {i}"
    assert "basal inputs (including CO2) at 10" in "".join(ab["cells"][17]["source"])
    assert "41-input AB-union checkpoint" in "".join(ab["cells"][23]["source"])


def test_ab_union_protocol_retains_regime_media(setup):
    model, outputs, _, _ = setup
    inputs = ev.build_input_columns()
    amn = ev.load_amn(ROOT / "AMN_data", inputs, "AB_union", feature_order="checkpoint")
    minn = ev.load_minn(ROOT / "MINN_data", outputs, input_names=inputs,
                        model_family="AB_union", legacy_protocol=True)
    assert amn["fixed"]["EX_pi_e"] == amn["fixed"]["EX_co2_e"] == 10
    assert amn["fixed"]["EX_glyc_e"] == 2.2
    assert all(amn["fixed"][x] == 0 for x in ["EX_glc__D_e", "EX_etoh_e", "EX_cbl1_e"])
    assert minn["fixed"]["EX_pi_e"] == minn["fixed"]["EX_cbl1_e"] == 50
    assert amn["features"] == [x for x in inputs if x in ev.AMN_CARBON_EXCHANGES + ["EX_o2_e"]]

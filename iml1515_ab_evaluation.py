"""Evaluation helpers for the A union B notebook; no training runs on import."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from flux_transformer import FluxTransformer
from generate_ecoli_iML1515_AB_union_data import (
    AMINO_EXCHANGES, AMN_CARBON_EXCHANGES, B_ONLY_BASE_EXCHANGES,
    COMMON_BASE_EXCHANGES, FIXED_CARBON_EXCHANGES, MINN_CONTEXT_EXCHANGES,
    build_input_columns,
)

OBJECTIVE = "BIOMASS_Ec_iML1515_core_75p37M"
CONTEXT_SOURCES = [
    "R_EX_glc__D_e_rev", "R_EX_o2_e_rev", "R_EX_co2_e_fwd",
    "R_EX_etoh_e", "R_EX_ac_e",
]
FLUX_FILES = {
    "minn_fitted": "fluxomics_iAF1260_reduced_split_fit.csv",
    "non_fitted": "fluxomics_iAF1260_reduced_split.csv",
    "iml1515_minn_like": "fluxomics_iML1515_minn_like_fit.csv",
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def load_reservoir(path, device, metadata_path=None):
    """Require explicit architecture/schema, including for temporary checkpoints."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Use a checkpoint containing model_state_dict, not bare weights.")
    metadata = checkpoint
    if metadata_path is not None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    config, info = metadata.get("config", {}), metadata.get("data_info", {})
    required = {"d_model", "n_heads", "n_layers", "d_ff", "dropout", "vocab_size"}
    if not required.issubset(config) or not {"input_cols", "output_cols"}.issubset(info):
        raise ValueError("Missing config/schema: supply verified training metadata JSON.")
    inputs, outputs = list(info["input_cols"]), list(info["output_cols"])
    if inputs != build_input_columns() or len(inputs) != 41:
        raise ValueError("Checkpoint input order does not match the 41-input A union B generator.")
    if len(outputs) != len(set(outputs)) or len(outputs) != config["vocab_size"]:
        raise ValueError("Invalid checkpoint output vocabulary.")
    indices = [outputs.index(f"{name}_flux") for name in inputs]
    state_indices = checkpoint["model_state_dict"]["input_token_indices"].tolist()
    if indices != state_indices:
        raise ValueError("Input-token indices disagree with the checkpoint schema.")
    for source in (config, info):
        if "input_token_indices" in source and list(source["input_token_indices"]) != indices:
            raise ValueError("Conflicting metadata input-token indices.")
    model = FluxTransformer(input_token_indices=indices, **{k: config[k] for k in required})
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, inputs, outputs, config, info


def metrics(truth, prediction):
    """Regression R2 and squared Pearson correlation are intentionally separate."""
    a, b = np.asarray(truth, dtype=float).ravel(), np.asarray(prediction, dtype=float).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Metrics require aligned finite arrays; handle failed samples explicitly.")
    residual = a - b
    denom = np.sum((a - a.mean()) ** 2)
    norm = np.linalg.norm(a)
    return {
        "R2": float(1 - np.sum(residual ** 2) / denom) if denom > 0 else np.nan,
        "Pearson_r2": float(np.corrcoef(a, b)[0, 1] ** 2)
        if a.size > 1 and np.std(a) > 0 and np.std(b) > 0 else np.nan,
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "NE": float(np.linalg.norm(residual) / norm) if norm > 0 else np.nan,
    }


def metric_tables(truth, prediction, names, ids):
    per_flux = pd.DataFrame([
        {"flux": name, **metrics(truth[:, i], prediction[:, i])}
        for i, name in enumerate(names)
    ])
    per_sample = pd.DataFrame([
        {"sample": sample, **metrics(a, b)} for sample, a, b in zip(ids, truth, prediction)
    ])
    return per_flux, per_sample


def summarize_amn_oof(oof, n_samples, split_seeds):
    """Match model C: score per-medium mean predictions; SD over repeat scores."""
    if set(oof["repeat"]) != set(split_seeds):
        raise ValueError("AMN OOF repeat seeds do not match configuration.")
    if oof.duplicated(["repeat", "row"]).any():
        raise ValueError("Duplicate AMN OOF sample within a repeat.")
    for _, part in oof.groupby("repeat"):
        if set(part["row"]) != set(range(n_samples)):
            raise ValueError("Incomplete AMN OOF repeat.")
    if not oof.groupby("row")["truth"].nunique().eq(1).all():
        raise ValueError("AMN truth changes between repeats.")
    means = oof.groupby("row", sort=True).agg(
        truth=("truth", "first"), prediction=("prediction", "mean"),
        prediction_std=("prediction", "std"))
    means["prediction_std"] = means["prediction_std"].fillna(0.0)
    scores = metrics(means.truth, means.prediction)
    repeat_scores = pd.DataFrame([metrics(part.truth, part.prediction)
                                 for _, part in oof.groupby("repeat")])
    spread = repeat_scores.std(ddof=0).to_dict()
    return means, scores, spread


def finite_frame(frame, label):
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains missing/nonfinite values; resolve explicitly.")


def load_amn(root):
    root = Path(root)
    frame = pd.read_csv(root / "iML1515_EXP.csv").rename(columns=lambda c: c.removesuffix("_i"))
    uncertainty = pd.read_csv(root / "EXP110.csv").rename(columns=lambda c: c.removesuffix("_i"))
    carbons = list(AMN_CARBON_EXCHANGES)
    finite_frame(frame, "AMN")
    if len(frame) != 110 or frame.columns.duplicated().any():
        raise ValueError("Expected 110 AMN rows and unique columns.")
    # Medium composition identifies these conditions independently of file row order.
    if frame.duplicated(carbons).any() or uncertainty.duplicated(carbons).any():
        raise ValueError("AMN medium keys are not unique.")
    joined = frame[carbons + ["GR_AVG"]].merge(
        uncertainty[carbons + ["GR_AVG", "GR_STD"]], on=carbons,
        how="left", validate="one_to_one", suffixes=("", "_uncertainty"),
    )
    finite_frame(joined, "AMN uncertainty join")
    if not np.allclose(joined.GR_AVG, joined.GR_AVG_uncertainty, atol=1e-6):
        raise ValueError("AMN growth values disagree after uncertainty alignment.")
    features = carbons + ["EX_o2_e"]
    if not np.isin(frame[features], [0, 1]).all():
        raise ValueError("Expected binary AMN presence features.")
    absent = ["EX_glc__D_e", "EX_etoh_e", "EX_cbl1_e"]
    for name in absent:
        if name in frame and frame[name].ne(0).any():
            raise ValueError(f"Unexpected A-regime presence: {name}")
    fixed = {name: 0.0 for name in build_input_columns()}
    for name in COMMON_BASE_EXCHANGES + ["EX_co2_e"] + FIXED_CARBON_EXCHANGES + AMINO_EXCHANGES:
        if frame[name].nunique() != 1 or frame[name].iloc[0] not in (0, 1):
            raise ValueError(f"Expected fixed AMN presence flag: {name}")
        rate = 2.2 if name in FIXED_CARBON_EXCHANGES + AMINO_EXCHANGES else 10.0
        fixed[name] = rate * float(frame[name].iloc[0])
    return {
        "X": frame[features].to_numpy(np.float32),
        "y": frame[["GR_AVG"]].to_numpy(np.float32),
        "observed": np.zeros((len(frame), 2), np.float32),
        "ids": np.array([f"medium_{i:03d}" for i in range(len(frame))]),
        "features": features, "targets": [f"{OBJECTIVE}_flux"], "fixed": fixed,
        "strata": frame[carbons].sum(axis=1).to_numpy(int),
        "growth_std": joined.GR_STD.to_numpy(float),
    }


def map_minn(source, outputs):
    if source.startswith("R_BIOMASS"):
        token, sign = f"{OBJECTIVE}_flux", 1.0
    else:
        name, sign = source.removeprefix("R_"), 1.0
        if name.endswith("_rev"):
            name, sign = name[:-4], -1.0
        elif name.endswith("_fwd"):
            name = name[:-4]
        token = f"{name}_flux"
    if token not in outputs:
        raise ValueError(f"Unmapped MINN source: {source} -> {token}")
    return token, sign


def load_minn(root, outputs, mode="minn_fitted"):
    root = Path(root)
    frames = [pd.read_csv(root / name).set_index("experiment") for name in
              ("transcriptomics.csv", "proteomics.csv", FLUX_FILES[mode])]
    transcript, protein, flux = frames
    for frame in frames:
        if len(frame) != 29 or not frame.index.is_unique or frame.columns.duplicated().any():
            raise ValueError("Expected 29 unique MINN experiments and unique feature names.")
        if set(frame.index) != set(transcript.index):
            raise ValueError("MINN experiment IDs differ across files.")
        finite_frame(frame, "MINN")
    flux = flux.loc[transcript.index]
    features = transcript.join(protein).join(flux[CONTEXT_SOURCES[:2]])
    mappings = [(src, *map_minn(src, outputs)) for src in flux.columns]
    if len({token for _, token, _ in mappings}) != len(mappings):
        raise ValueError("Duplicate mapped MINN targets.")
    target_map = [(s, t, sign) for s, t, sign in mappings if s not in CONTEXT_SOURCES]
    observed = flux[CONTEXT_SOURCES[:2]].to_numpy(np.float32)
    if (observed < 0).any():
        raise ValueError("Glucose/O2 inputs must be positive split uptake magnitudes.")
    return {
        "X": features.to_numpy(np.float32),
        "y": flux[[s for s, _, _ in target_map]].to_numpy(np.float32)
             * np.array([sign for _, _, sign in target_map], np.float32),
        "observed": observed, "ids": flux.index.to_numpy(),
        "features": features.columns.tolist(), "targets": [t for _, t, _ in target_map],
        "fixed": {name: 50.0 for name in COMMON_BASE_EXCHANGES + B_ONLY_BASE_EXCHANGES},
        "mapping": mappings, "flux": flux, "mode": mode,
        "context_truth": flux[CONTEXT_SOURCES].to_numpy(float),
    }


class FrontReservoir(nn.Module):
    """A fresh task-specific MLP, with one shared immutable reservoir."""

    def __init__(self, reservoir, outputs, data, task, mode="measured", dropout=0.0, hidden=512):
        super().__init__()
        if task not in ("amn", "minn") or mode not in ("measured", "predicted"):
            raise ValueError("Unknown task/context mode.")
        self.reservoir, self.task, self.mode = reservoir, task, mode
        self.reservoir.requires_grad_(False).eval()
        controlled = data["features"] if task == "amn" else list(MINN_CONTEXT_EXCHANGES)
        self.register_buffer("controlled", torch.tensor([outputs.index(f"{s}_flux") for s in controlled]))
        self.register_buffer("targets", torch.tensor([outputs.index(s) for s in data["targets"]]))
        fixed = torch.zeros(len(outputs))
        for name, value in data["fixed"].items():
            fixed[outputs.index(f"{name}_flux")] = value
        self.register_buffer("fixed", fixed)
        count = len(controlled) if task == "amn" or mode == "predicted" else 3
        self.front = nn.Sequential(nn.Linear(data["X"].shape[1], hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, count))
        self.register_buffer("scales", torch.tensor([10.0 if s == "EX_o2_e" else 2.2 for s in controlled]))

    def train(self, mode=True):
        super().train(mode)
        self.reservoir.eval()
        return self

    def medium(self, x, observed):
        raw = self.front(x)
        if self.task == "amn":
            values = torch.sigmoid(raw) * self.scales * (x > 0)
        else:
            values = nn.functional.softplus(raw)
            if self.mode == "measured":
                values = torch.cat([observed, values], dim=1)
        medium = self.fixed.unsqueeze(0).expand(len(x), -1).clone()
        medium[:, self.controlled] = values
        return medium, values

    def forward(self, x, observed):
        medium, context = self.medium(x, observed)
        prediction, _ = self.reservoir(medium.unsqueeze(-1), output_subset=None)
        return prediction[:, self.targets, -1], context


def predict_front(model, x, observed, batch_size=1):
    model.eval()
    device = next(model.front.parameters()).device
    predictions, contexts = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            pred, context = model(torch.as_tensor(x[start:start + batch_size], device=device),
                                  torch.as_tensor(observed[start:start + batch_size], device=device))
            predictions.append(pred.float().cpu().numpy())
            contexts.append(context.float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(contexts)


def fit_front(reservoir, outputs, data, task, mode, train_ids, settings, params,
              seed, validation_ids=None, epochs=None):
    """Retry an overflowing mixed-precision fit from its original seed in FP32."""
    try:
        return _fit_front(reservoir, outputs, data, task, mode, train_ids, settings,
                          params, seed, validation_ids, epochs)
    except FloatingPointError:
        if not settings["amp"] or next(reservoir.parameters()).device.type != "cuda":
            raise
    # Leave the exception handler before retrying so failed-fit tensors are released.
    torch.cuda.empty_cache()
    print(f"{task} {mode}: mixed-precision overflow; restarting this fit in FP32.", flush=True)
    return _fit_front(reservoir, outputs, data, task, mode, train_ids,
                      {**settings, "amp": False}, params, seed, validation_ids, epochs)


def _fit_front(reservoir, outputs, data, task, mode, train_ids, settings, params,
               seed, validation_ids=None, epochs=None):
    """Outer test rows are never passed here; refits have no validation targets."""
    seed_all(seed)
    train_ids = np.asarray(train_ids)
    if validation_ids is not None and np.intersect1d(train_ids, validation_ids).size:
        raise ValueError("Training/validation overlap.")
    device = next(reservoir.parameters()).device
    scaler = MinMaxScaler().fit(data["X"][train_ids]) if task == "minn" else None
    def transform(rows):
        values = scaler.transform(data["X"][rows]) if scaler else data["X"][rows]
        return values.astype(np.float32)
    model = FrontReservoir(reservoir, outputs, data, task, mode, params["drop_rate"], settings["hidden"]).to(device)
    optimizer = torch.optim.AdamW(model.front.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    criterion = nn.HuberLoss(delta=settings["delta"])
    max_epochs = settings["epochs"] if epochs is None else int(epochs)
    # Keep the same schedule horizon during inner selection and fixed-epoch refit.
    def lr_factor(epoch):
        warmup = settings["warmup"]
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = min(1.0, (epoch - warmup) / max(1, settings["epochs"] - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor) if task == "minn" else None
    x_train = transform(train_ids)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train),
        torch.from_numpy(data["y"][train_ids]), torch.from_numpy(data["observed"][train_ids]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)
    use_amp = settings["amp"] and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(max_epochs):
        model.train()
        train_loss, count = 0.0, 0
        for x, y, observed in loader:
            x, y, observed = x.to(device), y.to(device), observed.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                prediction, _ = model(x, observed)
                loss = criterion(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite front-network loss.")
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            try:
                nn.utils.clip_grad_norm_(model.front.parameters(), settings["clip"], error_if_nonfinite=True)
            except RuntimeError as exc:
                if "non-finite" not in str(exc):
                    raise
                raise FloatingPointError("Nonfinite front-network gradients; optimizer update rejected.") from exc
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_loss += loss.item() * len(x)
            count += len(x)
        if scheduler:
            scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": train_loss / count, "amp": use_amp}
        if validation_ids is not None:
            pred, _ = predict_front(model, transform(validation_ids), data["observed"][validation_ids], settings["batch_size"])
            val_loss = criterion(torch.from_numpy(pred), torch.from_numpy(data["y"][validation_ids])).item()
            row["validation_loss"] = val_loss
            if val_loss < best_loss - settings["min_delta"]:
                best_loss, best_epoch, stale = val_loss, epoch + 1, 0
                best_state = deepcopy(model.front.state_dict())
            else:
                stale += 1
        history.append(row)
        if validation_ids is not None and stale >= settings["patience"]:
            break
    if validation_ids is not None:
        if best_state is None:
            raise RuntimeError("No finite inner validation checkpoint.")
        model.front.load_state_dict(best_state)
    else:
        best_epoch = max_epochs
    return model, scaler, best_epoch, best_loss, history


def run_amn(reservoir, outputs, data, settings, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows, selection = [], []
    params = settings["params"]
    for repeat in settings["split_seeds"]:
        folds = StratifiedKFold(settings["folds"], shuffle=True, random_state=repeat)
        for fold, (train, test) in enumerate(folds.split(data["X"], data["strata"]), 1):
            inner_train, inner_val = train_test_split(train, test_size=settings["inner_fraction"],
                stratify=data["strata"][train], random_state=repeat + fold)
            selected, _, epochs, _, history = fit_front(reservoir, outputs, data, "amn", "measured",
                inner_train, settings, params, settings["train_seed"], validation_ids=inner_val)
            del selected
            model, _, _, _, refit_history = fit_front(reservoir, outputs, data, "amn", "measured",
                train, settings, params, settings["train_seed"], epochs=epochs)
            prediction, context = predict_front(model, data["X"][test], data["observed"][test], settings["batch_size"])
            prefix = destination / f"repeat_{repeat}_fold_{fold}"
            torch.save(model.front.state_dict(), str(prefix) + "_front.pth")
            save_json(str(prefix) + "_fold.json", {"train": train.tolist(), "test": test.tolist(),
                "inner_train": inner_train.tolist(), "inner_validation": inner_val.tolist(),
                "epochs": epochs, "params": params, "selection_history": history, "refit_history": refit_history})
            pd.DataFrame(context, index=data["ids"][test], columns=data["features"]).to_csv(str(prefix) + "_context.csv")
            selection.append({"repeat": repeat, "fold": fold, "epochs": epochs})
            for i, value in zip(test, prediction[:, 0]):
                rows.append({"sample": data["ids"][i], "row": int(i), "repeat": repeat, "fold": fold,
                    "truth": float(data["y"][i, 0]), "prediction": float(value), "carbon_count": int(data["strata"][i])})
            pd.DataFrame(rows).to_csv(destination / "oof.csv", index=False)
            print(f"AMN repeat {repeat}, fold {fold}: refit {epochs} epochs", flush=True)
            del model
    oof = pd.DataFrame(rows)
    if not oof.groupby(["repeat", "sample"]).size().eq(1).all() or len(oof) != len(data["X"]) * len(settings["split_seeds"]):
        raise AssertionError("Incomplete/duplicate AMN OOF predictions.")
    summary = pd.DataFrame([{"repeat": repeat, **metrics(part.truth, part.prediction)} for repeat, part in oof.groupby("repeat")])
    summary.to_csv(destination / "repeat_metrics.csv", index=False)
    pd.DataFrame(selection).to_csv(destination / "selected_epochs.csv", index=False)
    return {"oof": oof, "summary": summary}


def run_minn(reservoir, outputs, data, mode, settings, destination):
    import optuna
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = np.full_like(data["y"], np.nan)
    contexts = np.full((len(data["X"]), 5), np.nan)
    fold_records = []
    # Model C protocol: one full-dataset HPO study per context mode, then LOO.
    hpo_rows = np.arange(len(data["X"]))
    inner_splits = list(KFold(settings["inner_folds"], shuffle=True,
                             random_state=settings["seed"]).split(hpo_rows))
    def objective(trial):
        params = {"drop_rate": trial.suggest_categorical("drop_rate", settings["drop_rates"]),
            "learning_rate": trial.suggest_float("learning_rate", *settings["lr_range"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *settings["wd_range"], log=True)}
        losses, epochs = [], []
        for inner, (itr, iva) in enumerate(inner_splits):
            model, _, best_epoch, loss, _ = fit_front(reservoir, outputs, data, "minn", mode,
                hpo_rows[itr], settings, params, settings["seed"] + inner, validation_ids=hpo_rows[iva])
            losses.append(loss)
            epochs.append(best_epoch)
            del model
            running = float(np.mean(losses) + settings["std_penalty"] * (np.std(losses, ddof=1) if len(losses) > 1 else 0.0))
            trial.report(running, step=inner)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("inner_epochs", epochs)
        trial.set_user_attr("inner_losses", losses)
        return float(np.mean(losses) + settings["std_penalty"] * np.std(losses, ddof=1))
    study = optuna.create_study(direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=settings["seed"], multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(5, settings["inner_folds"]),
                                         n_warmup_steps=1, interval_steps=1))
    study.optimize(objective, n_trials=settings["trials"], catch=(FloatingPointError,))
    study.trials_dataframe().to_csv(destination / "global_hpo_trials.csv", index=False)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError("No MINN HPO trial completed with finite gradients/loss. Inspect the failed trials.")
    print(f"MINN {mode} global best parameters: {study.best_params}", flush=True)
    hpo_splits = [{"train": hpo_rows[a].tolist(), "validation": hpo_rows[b].tolist()}
                  for a, b in inner_splits]
    save_json(destination / "global_hpo.json", {"params": study.best_params,
        "objective": study.best_value, "splits": hpo_splits, "scope": "full_dataset"})
    for outer, (train, test) in enumerate(LeaveOneOut().split(data["X"])):
        # Match C for this run; changing outer-fold early stopping is deferred.
        model, scaler, epochs, _, history = fit_front(reservoir, outputs, data, "minn", mode,
            train, settings, study.best_params, settings["seed"], validation_ids=test)
        pred, context = predict_front(model, scaler.transform(data["X"][test]).astype(np.float32),
                                      data["observed"][test], settings["batch_size"])
        predictions[test], contexts[test] = pred, context
        record = {"fold": outer, "test": test.tolist(), "train": train.tolist(),
            "test_sample": str(data["ids"][test[0]]), "epochs": epochs, "params": study.best_params,
            "hpo_scope": "full_dataset", "early_stopping_scope": "outer_test",
            "hpo_splits": hpo_splits,
            "inner_epochs": study.best_trial.user_attrs["inner_epochs"],
            "scaler_min": scaler.min_.tolist(), "scaler_scale": scaler.scale_.tolist(), "history": history}
        fold_records.append(record)
        prefix = destination / f"fold_{outer:02d}"
        save_json(str(prefix) + ".json", record)
        torch.save(model.front.state_dict(), str(prefix) + "_front.pth")
        pd.DataFrame(predictions, index=data["ids"], columns=data["targets"]).to_csv(destination / "oof_predictions.csv")
        pd.DataFrame(contexts, index=data["ids"], columns=CONTEXT_SOURCES).to_csv(destination / "oof_context.csv")
        print(f"MINN {mode}: {outer + 1}/{len(data['X'])}, best epoch {epochs}", flush=True)
        del model
    if not np.isfinite(predictions).all() or not np.isfinite(contexts).all():
        raise AssertionError("Incomplete MINN OOF predictions.")
    if mode == "measured" and not np.allclose(contexts[:, :2], data["observed"]):
        raise AssertionError("Measured context was changed.")
    per_flux, per_sample = metric_tables(data["y"], predictions, data["targets"], data["ids"])
    per_flux.to_csv(destination / "per_flux.csv", index=False)
    per_sample.to_csv(destination / "per_sample.csv", index=False)
    pd.DataFrame(data["y"], index=data["ids"], columns=data["targets"]).to_csv(destination / "truth.csv")
    return {"mode": mode, "prediction": predictions, "context": contexts,
            "per_flux": per_flux, "per_sample": per_sample, "folds": fold_records,
            "pooled": metrics(data["y"], predictions)}


def run_pfba(xml_path, data, context=None, cap_mode="co2_etoh_ac_cap", fraction=0.999):
    from cobra.io import read_sbml_model
    from cobra.flux_analysis import pfba
    if cap_mode not in ("co2_etoh_ac_cap", "etoh_ac_cap"):
        raise ValueError("Unknown pFBA cap set.")
    if context is not None:
        context = np.asarray(context)
        if context.shape != (len(data["ids"]), 5) or not np.isfinite(context).all() or (context < 0).any():
            raise ValueError("Expected finite, nonnegative OOF context in aligned sample order.")
    model = read_sbml_model(str(xml_path))
    model.objective = OBJECTIVE
    mappings = data["mapping"]
    prediction = np.full((len(data["ids"]), len(mappings)), np.nan)
    statuses, diagnostics = [], []
    truth = data["flux"][[s for s, _, _ in mappings]].to_numpy(float)
    for i, sample in enumerate(data["ids"]):
        with model:
            # Preserve the maintained Table 4 SBML background medium for all methods.
            for j, name in enumerate(MINN_CONTEXT_EXCHANGES[:2]):
                model.reactions.get_by_id(name).lower_bound = -float(data["observed"][i, j])
            cap_positions = [2, 3, 4] if cap_mode == "co2_etoh_ac_cap" else [3, 4]
            if context is not None:
                for j in cap_positions:
                    rxn = model.reactions.get_by_id(MINN_CONTEXT_EXCHANGES[j])
                    cap = float(context[i, j])
                    rxn.bounds = (max(0.0, min(rxn.lower_bound, cap)), cap)
            try:
                solution = pfba(model, fraction_of_optimum=fraction)
                if solution.status != "optimal":
                    raise RuntimeError(solution.status)
                values = [float(solution.fluxes[token[:-5]]) * sign for _, token, sign in mappings]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite solution.")
                prediction[i] = values
                statuses.append({"sample": sample, "status": "optimal", "error": ""})
                if context is not None:
                    for j in cap_positions:
                        cap, flux = float(context[i, j]), float(solution.fluxes[MINN_CONTEXT_EXCHANGES[j]])
                        target = float(data["context_truth"][i, j])
                        binding = bool(abs(cap - flux) <= 1e-6)
                        diagnostics.append({"sample": sample, "source": CONTEXT_SOURCES[j],
                            "cap": cap, "flux": flux, "target": target, "slack": cap - flux,
                            "binding": binding, "binding_low_cap": binding and cap < target - 1e-6})
            except Exception as exc:
                statuses.append({"sample": sample, "status": "failed", "error": str(exc)})
    success = np.isfinite(prediction).all(axis=1)
    names = [s for s, _, _ in mappings]
    if success.any():
        per_flux, per_sample = metric_tables(truth[success], prediction[success], names, data["ids"][success])
    else:
        per_flux, per_sample = pd.DataFrame(), pd.DataFrame()
    return {"prediction": prediction, "truth": truth, "success": success, "names": names,
            "status": pd.DataFrame(statuses), "caps": pd.DataFrame(diagnostics),
            "per_flux": per_flux, "per_sample": per_sample}


def compare_pfba(results, data, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    common = np.logical_and.reduce([r["success"] for r in results.values()])
    rows = []
    for method, result in results.items():
        for name in ("status", "caps", "per_flux", "per_sample"):
            result[name].to_csv(destination / f"{method}_{name}.csv", index=False)
        pd.DataFrame(result["prediction"], index=data["ids"], columns=result["names"]).to_csv(destination / f"{method}_prediction.csv")
        for coverage, mask in [("own_success", result["success"]), ("common_success", common)]:
            for target_set in ("all", "non_context"):
                columns = np.array([target_set == "all" or name not in CONTEXT_SOURCES for name in result["names"]])
                per_sample = [metrics(a[columns], b[columns]) for a, b in zip(result["truth"][mask], result["prediction"][mask])]
                for metric in ("R2", "Pearson_r2", "MAE", "RMSE", "NE"):
                    values = np.array([r[metric] for r in per_sample])
                    valid = values[np.isfinite(values)]
                    rows.append({"method": method, "coverage": coverage, "targets": target_set,
                        "successes": int(mask.sum()), "total": len(mask), "metric": metric,
                        "defined": len(valid), "mean": valid.mean() if len(valid) else np.nan,
                        "std": valid.std(ddof=0) if len(valid) else np.nan})
        if method != "baseline" and not result["per_sample"].empty and not results["baseline"]["per_sample"].empty:
            paired = result["per_sample"].merge(results["baseline"]["per_sample"], on="sample", suffixes=("", "_baseline"))
            paired["delta_MAE"] = paired.MAE - paired.MAE_baseline
            paired.to_csv(destination / f"{method}_paired_errors.csv", index=False)
            if not result["caps"].empty:
                result["caps"].merge(paired[["sample", "delta_MAE"]], on="sample").to_csv(destination / f"{method}_cap_error_diagnostics.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(destination / "comparison.csv", index=False)
    return table


def simulated_fidelity(reservoir, inputs, outputs, specs, xml_path, destination,
                       batch_size=2, objective_samples=100, fraction=0.999):
    """Stream independent A/B test files; score every row, solve an explicit subset."""
    from cobra.io import read_sbml_model
    from cobra.util.array import create_stoichiometric_matrix
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    gem = read_sbml_model(str(xml_path))
    gem.objective = OBJECTIVE
    if outputs != [f"{r.id}_flux" for r in gem.reactions]:
        raise ValueError("GEM reaction order differs from checkpoint output order.")
    stoich = create_stoichiometric_matrix(gem, array_type="lil").tocsr()
    rxn_idx = {r.id: i for i, r in enumerate(gem.reactions)}
    injection = [outputs.index(f"{name}_flux") for name in inputs]
    exchange_indices = np.array([rxn_idx[r.id] for r in gem.exchanges])
    default_lower = np.array([r.lower_bound for r in gem.reactions])
    default_upper = np.array([r.upper_bound for r in gem.reactions])
    device = next(reservoir.parameters()).device
    reservoir.eval()
    summary, physics = [], []
    combined = None
    for domain, spec in specs.items():
        if domain not in ("A", "B"):
            raise ValueError("A/B bound reconstruction only; additional domains need their own contract.")
        if spec["seed"] == 42:
            raise ValueError("Held-out seed must differ from training seed 42.")
        header = pd.read_csv(spec["path"], nrows=0).columns.tolist()
        if [c for c in header if c.endswith("_flux")] != outputs:
            raise ValueError(f"{domain}: simulated output order differs from checkpoint.")
        required = (COMMON_BASE_EXCHANGES + ["EX_co2_e", "EX_o2_e"]
                    + AMN_CARBON_EXCHANGES + FIXED_CARBON_EXCHANGES + AMINO_EXCHANGES
                    if domain == "A" else COMMON_BASE_EXCHANGES + B_ONLY_BASE_EXCHANGES + MINN_CONTEXT_EXCHANGES)
        if not set(required).issubset(header):
            raise ValueError(f"{domain}: missing required source input columns.")
        stats = {k: np.zeros(len(outputs), dtype=np.float64) for k in ("sum", "sum2", "sse", "abs", "active")}
        count = 0
        for frame in pd.read_csv(spec["path"], chunksize=128):
            truth = frame[outputs].to_numpy(np.float64)
            medium = np.column_stack([frame[name].to_numpy(float) if name in frame else np.zeros(len(frame)) for name in inputs])
            if not np.isfinite(truth).all() or not np.isfinite(medium).all() or (medium < 0).any():
                raise ValueError("Invalid simulated data.")
            forbidden = (["EX_glc__D_e", "EX_etoh_e", "EX_cbl1_e"] if domain == "A" else
                         list(set(AMN_CARBON_EXCHANGES + FIXED_CARBON_EXCHANGES + AMINO_EXCHANGES) - {"EX_ac_e"}))
            if any(np.any(medium[:, inputs.index(name)] != 0) for name in forbidden):
                raise ValueError(f"{domain}: test data contains inputs from the other regime.")
            lower = np.tile(default_lower, (len(frame), 1))
            upper = np.tile(default_upper, (len(frame), 1))
            if domain == "A":
                lower[:, exchange_indices] = 0
                upper[:, exchange_indices] = np.maximum(0, default_upper[exchange_indices])
            for name in set(required):
                pos = rxn_idx[name]
                if domain == "B" and name in MINN_CONTEXT_EXCHANGES[2:]:
                    lower[:, pos], upper[:, pos] = 0, medium[:, inputs.index(name)]
                else:
                    lower[:, pos] = -medium[:, inputs.index(name)]
            prediction = []
            with torch.no_grad():
                for start in range(0, len(frame), batch_size):
                    chunk = torch.zeros(min(batch_size, len(frame) - start), len(outputs), 1, device=device)
                    chunk[:, injection, 0] = torch.as_tensor(medium[start:start + batch_size], dtype=torch.float32, device=device)
                    pred, _ = reservoir(chunk, output_subset=None)
                    prediction.append(pred[:, :, -1].float().cpu().numpy())
            prediction = np.concatenate(prediction).astype(np.float64)
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite simulated predictions.")
            error = prediction - truth
            stats["sum"] += truth.sum(axis=0)
            stats["sum2"] += np.square(truth).sum(axis=0)
            stats["sse"] += np.square(error).sum(axis=0)
            stats["abs"] += np.abs(error).sum(axis=0)
            stats["active"] += (np.abs(truth) > 1e-8).sum(axis=0)
            residual = np.max(np.abs(stoich @ prediction.T), axis=0)
            violation = np.maximum(np.maximum(lower - prediction, prediction - upper), 0)
            for local in range(len(frame)):
                row = {"domain": domain, "row": count + local, "mass_balance_max": residual[local],
                    "bound_violation_max": violation[local].max(), "bound_violation_count": int((violation[local] > 1e-6).sum()),
                    "biomass_error": error[local, rxn_idx[OBJECTIVE]], "objective_status": "not_sampled"}
                if count + local < objective_samples:
                    with gem:
                        for j, reaction in enumerate(gem.reactions):
                            reaction.bounds = (float(lower[local, j]), float(upper[local, j]))
                        optimum = gem.slim_optimize(error_value=np.nan)
                    row["objective_status"] = "optimal" if np.isfinite(optimum) else "failed"
                    row["optimal_biomass"] = optimum
                    row["prediction_minus_pfba_objective"] = prediction[local, rxn_idx[OBJECTIVE]] - fraction * optimum
                physics.append(row)
            count += len(frame)
        if count == 0 or count != spec["rows"]:
            raise ValueError(f"{domain}: expected {spec['rows']} rows, found {count}.")
        def summarize(values, n, label):
            centered = values["sum2"] - np.square(values["sum"]) / n
            r2 = np.divide(values["sse"], centered, out=np.full(len(outputs), np.nan), where=centered > 1e-12)
            per_flux = pd.DataFrame({"flux": outputs, "R2": 1 - r2,
                "MAE": values["abs"] / n, "RMSE": np.sqrt(values["sse"] / n), "activity_frequency": values["active"] / n})
            per_flux.to_csv(destination / f"{label}_per_flux.csv", index=False)
            total = n * len(outputs)
            denominator = values["sum2"].sum() - values["sum"].sum() ** 2 / total
            return {"domain": label, "rows": n, "R2": 1 - values["sse"].sum() / denominator if denominator > 0 else np.nan,
                    "MAE": values["abs"].sum() / total, "RMSE": np.sqrt(values["sse"].sum() / total)}
        summary.append(summarize(stats, count, domain))
        if combined is None:
            combined = {k: v.copy() for k, v in stats.items()}
            combined_count = count
        else:
            for key in combined:
                combined[key] += stats[key]
            combined_count += count
        print(f"Simulated {domain}: {count} rows scored", flush=True)
    if len(specs) == 2:
        if specs["A"]["rows"] != specs["B"]["rows"]:
            raise ValueError("Balanced pooled score requires equal A/B row counts.")
        summary.append(summarize(combined, combined_count, "balanced_union"))
    pd.DataFrame(physics).to_csv(destination / "physical_diagnostics.csv", index=False)
    result = pd.DataFrame(summary)
    result.to_csv(destination / "summary.csv", index=False)
    return result

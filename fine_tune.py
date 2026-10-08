"""Fine-tuning del DKT-Forget de producción EDUFIN con interacciones reales.

Flujo:
1. Lee datasets/pilot_real_v1.csv.
2. Verifica un mínimo de usuarios/interacciones elegibles.
3. Separa train/validation POR USUARIO para evitar fuga de información.
4. Carga el checkpoint preentrenado existente.
5. Continúa el entrenamiento con learning rate bajo.
6. Evalúa modelo base y candidato sobre los mismos usuarios de validación.
7. Guarda un checkpoint candidato; NO reemplaza el modelo productivo.

El CSV debe contener:
    user_id, skill_id, correct, timestamp

Uso normal:
    python fine_tune.py

Opciones útiles:
    python fine_tune.py --min-users 10 --min-interactions 500
    python fine_tune.py --epochs 15 --learning-rate 0.0003
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dkt_forget_final import (
    DKTForgetDataset,
    DKTForgetSample,
    _next_step_vectors,
    collate_batch,
    evaluate_model,
    load_checkpoint,
    set_seed,
)


DEFAULT_DATASET = Path("datasets/pilot_real_v1.csv")
DEFAULT_BASE_CHECKPOINT = Path("checkpoints/dkt_forget_pretrained_v1.pth")
DEFAULT_OUTPUT = Path("checkpoints/candidates/dkt_forget_candidate_v2.pth")
DEFAULT_METADATA = Path("checkpoints/candidates/dkt_forget_candidate_v2.json")


# -----------------------------------------------------------------------------
# Utilidades de datos
# -----------------------------------------------------------------------------


def load_and_validate_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset real: {path}")

    df = pd.read_csv(path)
    required = ["user_id", "skill_id", "correct", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df[required].copy()
    df = df.dropna(subset=required)

    df["skill_id"] = pd.to_numeric(df["skill_id"], errors="raise").astype(int)
    invalid_skills = sorted(set(df.loc[~df["skill_id"].between(1, 30), "skill_id"].tolist()))
    if invalid_skills:
        raise ValueError(f"Hay skill_id fuera de 1..30: {invalid_skills}")

    df["correct"] = pd.to_numeric(df["correct"], errors="raise").astype(int)
    if not df["correct"].isin([0, 1]).all():
        raise ValueError("La columna correct solo puede contener 0 o 1.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True, errors="raise")

    # Eliminar duplicados exactos por seguridad.
    before = len(df)
    df = df.drop_duplicates(subset=required, keep="first")
    removed = before - len(df)
    if removed:
        print(f"Aviso: se eliminaron {removed} filas duplicadas exactas.")

    df = df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    return df


def filter_eligible_users(df: pd.DataFrame, min_interactions_per_user: int) -> Tuple[pd.DataFrame, pd.Series]:
    counts = df.groupby("user_id").size()
    eligible_ids = counts[counts >= min_interactions_per_user].index
    eligible = df[df["user_id"].isin(eligible_ids)].copy()
    return eligible, counts


def split_users(
    df: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    users = np.asarray(sorted(df["user_id"].astype(str).unique().tolist()))
    if len(users) < 2:
        raise ValueError("Se requieren al menos 2 usuarios elegibles para separar train/validation.")

    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    val_size = max(1, int(round(len(users) * validation_fraction)))
    if val_size >= len(users):
        val_size = len(users) - 1

    val_users = users[:val_size].tolist()
    train_users = users[val_size:].tolist()

    train_df = df[df["user_id"].astype(str).isin(train_users)].copy()
    val_df = df[df["user_id"].astype(str).isin(val_users)].copy()

    return train_df, val_df, train_users, val_users


def write_split_csvs(train_df: pd.DataFrame, val_df: pd.DataFrame, workdir: Path) -> Tuple[Path, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    train_path = workdir / "real_train.csv"
    val_path = workdir / "real_val.csv"

    # DKTForgetDataset acepta ISO-8601. Se escribe en UTC explícito.
    train_out = train_df.copy()
    val_out = val_df.copy()
    train_out["timestamp"] = train_out["timestamp"].map(lambda x: x.isoformat())
    val_out["timestamp"] = val_out["timestamp"].map(lambda x: x.isoformat())

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    return train_path, val_path


# -----------------------------------------------------------------------------
# Compatibilidad con las dimensiones temporales del checkpoint base
# -----------------------------------------------------------------------------


def clamp_temporal_features(
    dataset: DKTForgetDataset,
    *,
    num_rgap: int,
    num_sgap: int,
    num_pcount: int,
) -> Dict[str, int]:
    """Recorta bins temporales nuevos al máximo conocido por el checkpoint.

    El modelo preentrenado tiene dimensiones fijas para rgap/sgap/pcount.
    Interacciones reales posteriores pueden producir bins log2 mayores que los
    observados durante el pretraining. Para poder continuar con los mismos pesos,
    esos valores se saturan en la última categoría disponible.
    """
    if min(num_rgap, num_sgap, num_pcount) <= 0:
        raise ValueError("Dimensiones temporales inválidas en el checkpoint base.")

    clipped = {"rgap": 0, "sgap": 0, "pcount": 0}
    new_samples: List[DKTForgetSample] = []

    for sample in dataset.samples:
        rgaps = sample.rgaps.clone()
        sgaps = sample.sgaps.clone()
        pcounts = sample.pcounts.clone()

        clipped["rgap"] += int((rgaps >= num_rgap).sum().item())
        clipped["sgap"] += int((sgaps >= num_sgap).sum().item())
        clipped["pcount"] += int((pcounts >= num_pcount).sum().item())

        rgaps.clamp_(max=num_rgap - 1)
        sgaps.clamp_(max=num_sgap - 1)
        pcounts.clamp_(max=num_pcount - 1)

        new_samples.append(
            DKTForgetSample(
                skills=sample.skills,
                corrects=sample.corrects,
                rgaps=rgaps,
                sgaps=sgaps,
                pcounts=pcounts,
            )
        )

    dataset.samples = new_samples
    dataset.num_rgap = num_rgap
    dataset.num_sgap = num_sgap
    dataset.num_pcount = num_pcount
    return clipped


def build_dataset_and_loader(
    csv_path: Path,
    checkpoint: Dict[str, object],
    *,
    min_interactions: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Tuple[DKTForgetDataset, DataLoader, Dict[str, int]]:
    dataset = DKTForgetDataset(
        csv_path,
        min_interactions=min_interactions,
        skill2idx=checkpoint["skill2idx"],
        expected_num_skills=int(checkpoint["num_skills"]),
    )

    clipped = clamp_temporal_features(
        dataset,
        num_rgap=int(checkpoint["num_rgap"]),
        num_sgap=int(checkpoint["num_sgap"]),
        num_pcount=int(checkpoint["num_pcount"]),
    )

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        generator=generator if shuffle else None,
    )
    return dataset, loader, clipped


# -----------------------------------------------------------------------------
# Fine-tuning
# -----------------------------------------------------------------------------


def fine_tune(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    learning_rate: float,
    epochs: int,
    patience: int,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]], int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    best_score = -float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            preds = model(
                batch["skills"],
                batch["corrects"],
                batch["rgaps"],
                batch["sgaps"],
                batch["pcounts"],
            )
            selected, targets = _next_step_vectors(
                preds,
                batch["skills"],
                batch["corrects"],
                batch["mask"],
            )

            if selected.numel() == 0:
                continue

            loss = criterion(selected, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        if batches == 0:
            raise ValueError("El conjunto de training no produjo targets next-step válidos.")

        metrics = evaluate_model(model, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": running_loss / batches,
            "val_auc": float(metrics["auc"]),
            "val_acc": float(metrics["acc"]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={row['loss']:.4f} | "
            f"val_auc={row['val_auc']:.4f} | "
            f"val_acc={row['val_acc']:.4f}"
        )

        score = row["val_auc"] if not math.isnan(row["val_auc"]) else row["val_acc"]
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping: {patience} épocas sin mejora.")
            break

    return best_state, history, best_epoch


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning real del DKT-Forget EDUFIN")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--min-users", type=int, default=10)
    parser.add_argument("--min-interactions", type=int, default=500)
    parser.add_argument("--min-interactions-per-user", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not (0.0 < args.validation_fraction < 1.0):
        raise ValueError("validation-fraction debe estar entre 0 y 1.")
    if args.min_users < 2:
        raise ValueError("min-users debe ser al menos 2.")
    if args.min_interactions < 1:
        raise ValueError("min-interactions debe ser positivo.")

    print("Fine-tuning EDUFIN DKT-Forget")
    print("----------------------------------------")
    print(f"Dataset:           {args.dataset}")
    print(f"Checkpoint base:   {args.base_checkpoint}")

    df = load_and_validate_csv(args.dataset)
    eligible_df, user_counts = filter_eligible_users(df, args.min_interactions_per_user)

    total_users = int(df["user_id"].nunique())
    eligible_users = int(eligible_df["user_id"].nunique())
    eligible_interactions = int(len(eligible_df))

    print(f"Usuarios totales:               {total_users}")
    print(f"Usuarios elegibles (>= {args.min_interactions_per_user}): {eligible_users}")
    print(f"Interacciones elegibles:        {eligible_interactions}")

    reasons = []
    if eligible_users < args.min_users:
        reasons.append(f"usuarios insuficientes: {eligible_users}/{args.min_users}")
    if eligible_interactions < args.min_interactions:
        reasons.append(f"interacciones insuficientes: {eligible_interactions}/{args.min_interactions}")

    if reasons:
        print("\nFine-tuning omitido.")
        for reason in reasons:
            print(f"- {reason}")
        print("No se creó ningún checkpoint candidato.")
        return

    if not args.base_checkpoint.exists():
        raise FileNotFoundError(f"No existe el checkpoint base: {args.base_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, base_checkpoint = load_checkpoint(args.base_checkpoint, device=device)

    train_df, val_df, train_users, val_users = split_users(
        eligible_df,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    print("\nSplit por usuario")
    print(f"Train users:       {len(train_users)}")
    print(f"Validation users:  {len(val_users)}")
    print(f"Train rows:        {len(train_df)}")
    print(f"Validation rows:   {len(val_df)}")

    workdir = args.output.parent / "_fine_tune_tmp"
    train_csv, val_csv = write_split_csvs(train_df, val_df, workdir)

    _, train_loader, train_clipped = build_dataset_and_loader(
        train_csv,
        base_checkpoint,
        min_interactions=args.min_interactions_per_user,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    _, val_loader, val_clipped = build_dataset_and_loader(
        val_csv,
        base_checkpoint,
        min_interactions=args.min_interactions_per_user,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    clipped_total = {
        k: train_clipped[k] + val_clipped[k]
        for k in train_clipped
    }
    if any(clipped_total.values()):
        print("\nAviso: bins temporales saturados para conservar compatibilidad con el checkpoint:")
        for key, value in clipped_total.items():
            print(f"- {key}: {value}")

    base_metrics = evaluate_model(base_model, val_loader, device)
    print("\nModelo base sobre validation real")
    print(f"AUC:      {base_metrics['auc']:.4f}")
    print(f"Accuracy: {base_metrics['acc']:.4f}")

    candidate_model = copy.deepcopy(base_model).to(device)

    print("\nIniciando fine-tuning...")
    best_state, history, best_epoch = fine_tune(
        candidate_model,
        train_loader,
        val_loader,
        device,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
    )

    candidate_model.load_state_dict(best_state)
    candidate_model.to(device)
    candidate_model.eval()
    candidate_metrics = evaluate_model(candidate_model, val_loader, device)

    print("\nCandidato sobre validation real")
    print(f"Best epoch: {best_epoch}")
    print(f"AUC:        {candidate_metrics['auc']:.4f}")
    print(f"Accuracy:   {candidate_metrics['acc']:.4f}")

    auc_delta = (
        float(candidate_metrics["auc"] - base_metrics["auc"])
        if not math.isnan(candidate_metrics["auc"]) and not math.isnan(base_metrics["auc"])
        else float("nan")
    )
    acc_delta = float(candidate_metrics["acc"] - base_metrics["acc"])

    print("\nCambio candidato - base")
    print(f"Delta AUC:      {auc_delta:+.4f}" if not math.isnan(auc_delta) else "Delta AUC:      n/a")
    print(f"Delta Accuracy: {acc_delta:+.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    candidate_checkpoint = dict(base_checkpoint)
    candidate_checkpoint["state_dict"] = best_state
    candidate_checkpoint["parent_checkpoint"] = str(args.base_checkpoint)
    candidate_checkpoint["fine_tuning"] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "eligible_users": eligible_users,
        "eligible_interactions": eligible_interactions,
        "train_users": len(train_users),
        "validation_users": len(val_users),
        "train_interactions": int(len(train_df)),
        "validation_interactions": int(len(val_df)),
        "learning_rate": args.learning_rate,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "min_interactions_per_user": args.min_interactions_per_user,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "base_val_auc": float(base_metrics["auc"]),
        "base_val_acc": float(base_metrics["acc"]),
        "candidate_val_auc": float(candidate_metrics["auc"]),
        "candidate_val_acc": float(candidate_metrics["acc"]),
        "delta_auc": auc_delta,
        "delta_acc": acc_delta,
        "temporal_bins_clipped": clipped_total,
        "history": history,
    }

    torch.save(candidate_checkpoint, args.output)

    metadata = {
        "status": "candidate_only",
        "candidate_checkpoint": str(args.output),
        "base_checkpoint": str(args.base_checkpoint),
        "trained_at_utc": candidate_checkpoint["fine_tuning"]["trained_at_utc"],
        "best_epoch": best_epoch,
        "base_metrics": {
            "auc": float(base_metrics["auc"]),
            "accuracy": float(base_metrics["acc"]),
        },
        "candidate_metrics": {
            "auc": float(candidate_metrics["auc"]),
            "accuracy": float(candidate_metrics["acc"]),
        },
        "delta": {
            "auc": auc_delta,
            "accuracy": acc_delta,
        },
        "data": {
            "users": eligible_users,
            "interactions": eligible_interactions,
            "train_users": len(train_users),
            "validation_users": len(val_users),
        },
        "note": "Este archivo es un candidato. No se promueve automáticamente a producción.",
    }

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_output.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2, allow_nan=True)

    print("\nFine-tuning completado")
    print(f"Checkpoint candidato: {args.output}")
    print(f"Metadata:              {args.metadata_output}")
    print("IMPORTANTE: el candidato NO reemplazó el modelo productivo.")


if __name__ == "__main__":
    main()
